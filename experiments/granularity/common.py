"""Setup and helpers shared by the query-normalisation, lexical, kNN and control
experiments (query_norm, lexical, knn, controls).

``build_population`` fixes, without any embedding, the evaluation population of
run_corrections (val and test users with an H2 profile and at least one
categorised exact item), the train popularity and the non-personalised lists
(popularity, top category), and the cold-item and warm-item subsets of the exact
ground truth: an item is COLD when no TRAIN user's ground truth contains it.
``load_embeddings`` adds the matrices, the H2 profiles and the fitted moments of
the three statistics sources of the query-normalisation family. The scoring
helpers wrap ``run_corrections.score_split`` so every experiment reports the
same metrics, and the paired-test helpers build the committed statistics tables.
"""

from __future__ import annotations

import json
import logging
import math
import resource
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from acw.corrections import Moments, fit_moments
from acw.data.splits import SPLIT_NAMES, load_split, split_path
from acw.embeddings.generate_h2_profiles.io import load_user_history, user_history_path
from acw.evaluation.config import DomainConfig, resolve_paths
from acw.evaluation.loader import build_ground_truth, item_genres
from acw.evaluation.pipeline import train_popularity, trivial_rankings
from acw.evaluation.ranking import top_k_scores, top_k_scores_ties
from acw.evaluation.significance import add_holm_columns, paired_row
from acw.provenance import provenance
from acw.settings import GRANULARITY_DIR, PROCESSED_DIR
from experiments.granularity.hybrid import INF_SCALE, LAMBDAS, lam_label, popularity_z
from experiments.granularity.run_corrections import (
    METRICS,
    REGISTRY,
    UNVERIFIED_ENCODERS,
    Source,
    _load_matrix,
    h2_profiles,
    score_split,
    window_books,
)

log = logging.getLogger("granularity")

OUT_DIR = GRANULARITY_DIR  # results/granularity/, or seed{S}/ under ACW_SPLIT_SEED=S
DOMAINS = ("music", "movies", "games")
ITEM_TEXTS = ("titlecat", "desc", "title")  # item text variants of run_corrections (--text)
SOURCES = ("train_users", "books_catalog", "target_catalog")
N_BOOT = 2_000
N_FLIP = 10_000
ALPHA = 0.05
STAT_METRICS = {"nDCG@10": "ndcg10", "R@100": "r100"}


def text_tag(text: str) -> str:
    """Suffix of the result files of an item-text variant: none for the title+categories text
    of the first runs, ``_<text>`` for the others (``query_norm_music_gte_title``)."""
    return "" if text == "titlecat" else f"_{text}"


def verified_runs() -> list[tuple[str, str, str]]:
    """(domain, encoder, text) of the runs of the query-normalisation family whose embeddings
    passed the audit of docs/EMBEDDINGS.md: the title+categories text of gte in every domain and
    of every verified music encoder, and every other text of music/gte."""
    return [
        (d, e, t)
        for (d, e, t) in REGISTRY
        if e not in UNVERIFIED_ENCODERS
        and (d == "music" or e == "gte")
        and (t == "titlecat" or (d, e) == ("music", "gte"))
    ]


def gte_runs() -> list[tuple[str, str]]:
    """(domain, text) of the gte runs of lexical, kNN and controls: the three domains with the
    title+categories text, then the other texts of the REGISTRY (a missing result is skipped)."""
    return [(d, "titlecat") for d in DOMAINS] + [
        (d, t) for (d, e, t) in REGISTRY if e == "gte" and t != "titlecat"
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Population (no embeddings)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Population:
    domain: str
    cfg: DomainConfig
    history: dict[str, list[str]]
    train_users: list[str]
    train_kept: list[str]  # train users with an H2 profile
    eval_users: dict[str, list[str]]  # val / test
    exact_gt: dict[str, set[str]]
    raw_gt: dict[str, list[str]]  # exact interactions of every user (catalog or not)
    item_idx: dict[str, int]
    book_idx: dict[str, int]
    idx_to_item: dict[int, str]
    popularity: dict[str, int]  # number of train users per item
    counts: np.ndarray  # the same, per catalog row
    pz: np.ndarray  # z-score over the catalog of log(1 + popularity)
    tie_rank: np.ndarray  # rank of the item id among the catalog (ties are broken by it)
    constants: dict[str, dict[str, list[str]]]  # mode -> popularity / top_category lists
    strata: dict[str, dict[str, set[str]]]  # cold / warm relevant sets of every eval user
    inputs: list[Path]  # files read, for the provenance record


def load_ids(src: Source, cfg: DomainConfig) -> tuple[dict[str, int], dict[str, int]]:
    """Item and book id maps of an encoder, checked against the domain catalog and the
    canonical book catalog (as run_corrections does)."""
    with open(PROCESSED_DIR / src.items_idx) as f:
        item_idx = json.load(f)
    with open(PROCESSED_DIR / src.books_idx) as f:
        book_idx = json.load(f)
    with open(cfg.items_idx_path) as f:
        if set(json.load(f)) != set(item_idx):
            raise ValueError("encoder item ids differ from the domain catalog")
    with open(PROCESSED_DIR / "book_id_to_idx.json") as f:
        if set(json.load(f)) != set(book_idx):
            raise ValueError("encoder book ids differ from the canonical book catalog")
    return item_idx, book_idx


def build_population(domain: str, src: Source) -> Population:
    cfg = resolve_paths(domain)
    item_idx, book_idx = load_ids(src, cfg)
    idx_to_item = {v: k for k, v in item_idx.items()}
    history = load_user_history(domain)
    train_users = load_split(domain, "train")
    exact_gt = build_ground_truth(cfg, "exact")
    categorised = set(item_genres(cfg, 1))
    with open(cfg.gt_path) as f:
        raw_gt = json.load(f)

    def has_profile(u: str) -> bool:
        return bool(window_books(history.get(u, []), book_idx))

    eval_users = {}
    for s in ("val", "test"):
        kept = [u for u in load_split(domain, s) if has_profile(u)]
        eval_users[s] = [u for u in kept if exact_gt.get(u, set()) & categorised]
    popularity = train_popularity(cfg, train_users)
    n = len(idx_to_item)
    ids = np.array([idx_to_item[i] for i in range(n)])
    counts = np.array([popularity.get(idx_to_item[i], 0) for i in range(n)])
    pz = popularity_z(counts)
    by_depth = {
        d: trivial_rankings(cfg, idx_to_item, train_users, depth=d)[0]
        for d in {cfg.category_depth, 2}
    }
    base = by_depth[cfg.category_depth]
    constants = {"exact": base, "d2": by_depth[2], "cold": base, "warm": base}
    strata = {"cold": {}, "warm": {}}
    for s in eval_users.values():
        for u in s:
            rel = exact_gt[u]
            strata["cold"][u] = {i for i in rel if popularity.get(i, 0) == 0}
            strata["warm"][u] = rel - strata["cold"][u]
    inputs = [
        cfg.items_idx_path,
        cfg.items_meta_path,
        cfg.gt_path,
        user_history_path(domain),
        PROCESSED_DIR / src.items_idx,
        PROCESSED_DIR / src.books_idx,
        PROCESSED_DIR / "book_id_to_idx.json",
        *[split_path(domain, s) for s in SPLIT_NAMES],
    ]
    log.info(
        f"{domain}: eval users { {s: len(u) for s, u in eval_users.items()} }, "
        f"{len(train_users)} train users, {n} items"
    )
    return Population(
        domain, cfg, history, train_users,
        [u for u in train_users if has_profile(u)],
        eval_users, exact_gt, raw_gt, item_idx, book_idx, idx_to_item, popularity, counts, pz,
        np.argsort(np.argsort(ids)), constants, strata, inputs,
    )  # fmt: skip


# ─────────────────────────────────────────────────────────────────────────────
# Embeddings, profiles, moments
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Embeddings:
    items: np.ndarray
    train_prof: np.ndarray
    eval_prof: dict[str, np.ndarray]
    moments: dict[str, Moments]  # statistics source -> fitted moments
    shrunk: dict[str, Moments]  # same with Ledoit-Wolf shrunk eigenvalues
    shrinkage: dict[str, float]  # Ledoit-Wolf intensity of each source
    files: list[Path]


def shrink(mom: Moments, X: np.ndarray) -> tuple[Moments, float]:
    """Ledoit-Wolf shrinkage of the covariance of ``X`` towards ``mu * I`` (mu = mean
    eigenvalue): the intensity is sklearn's and the shrunk eigenvalues are
    ``(1 - delta) * lambda + delta * mu`` (same eigenvectors, same mean)."""
    from sklearn.covariance import ledoit_wolf_shrinkage

    delta = float(ledoit_wolf_shrinkage(X))
    lam = (1.0 - delta) * mom.eigvals + delta * mom.eigvals.mean()
    return Moments(mom.mean, lam, mom.eigvecs), delta


def load_embeddings(
    src: Source, pop: Population, sources: tuple[str, ...] = SOURCES, shrinkage: bool = True
) -> Embeddings:
    items, item_idx = _load_matrix(src.items, src.items_idx)
    books, book_idx = _load_matrix(src.books, src.books_idx)
    assert item_idx == pop.item_idx and book_idx == pop.book_idx
    train_prof, kept, _ = h2_profiles(pop.train_users, pop.history, books, book_idx)
    assert kept == pop.train_kept
    eval_prof = {}
    for s, users in pop.eval_users.items():
        eval_prof[s], kept, _ = h2_profiles(users, pop.history, books, book_idx)
        assert kept == users
    data = {"train_users": train_prof, "books_catalog": books, "target_catalog": items}
    moments, shrunk, delta = {}, {}, {}
    for name in sources:
        moments[name] = fit_moments(data[name])
        if shrinkage:
            shrunk[name], delta[name] = shrink(moments[name], data[name])
    files = [PROCESSED_DIR / f for f in (src.items, src.items_idx, src.books, src.books_idx)]
    return Embeddings(items, train_prof, eval_prof, moments, shrunk, delta, files)


# ─────────────────────────────────────────────────────────────────────────────
# Scoring and summaries
# ─────────────────────────────────────────────────────────────────────────────


def score(
    pop: Population,
    split: str,
    rankings: dict[str, np.ndarray],
    *,
    d2: bool = False,
    strata: bool = False,
) -> pd.DataFrame:
    """Per-user metrics of ``rankings`` (condition -> top-k item indices of the users of
    ``split``), the random baseline and the constant lists, under the exact ground truth
    plus, optionally, depth-2 categories and the cold / warm subsets (columns of
    ``run_corrections.score_split``)."""
    frames, *_ = score_split(
        split,
        pop.eval_users[split],
        rankings,
        pop.constants,
        {"d2": 2} if d2 else {},
        None,
        pop.exact_gt,
        pop.cfg,
        pop.idx_to_item,
        True,
        extra_gt=pop.strata if strata else None,
    )
    return pd.concat(frames, ignore_index=True)


def block_means(df: pd.DataFrame) -> dict:
    """gt -> condition -> means of the metrics, number of users and of users with a hit
    (nDCG@10 > 0 and Recall@100 > 0)."""
    d = df.assign(hit10=df["nDCG@10"] > 0, hit100=df["R@100"] > 0)
    g = d.groupby(["gt", "condition"], sort=False)
    m, n = g[METRICS].mean(), g.size()
    h10, h100 = g["hit10"].sum(), g["hit100"].sum()
    out: dict = {}
    for gt, c in m.index:
        out.setdefault(gt, {})[c] = {
            **{k: float(m.loc[(gt, c), k]) for k in METRICS},
            "n": int(n.loc[(gt, c)]),
            "n_hit10": int(h10.loc[(gt, c)]),
            "n_hit100": int(h100.loc[(gt, c)]),
        }
    return out


def exact_ndcg(val: pd.DataFrame, cond: str) -> float:
    return float(val.loc[(val["gt"] == "exact") & (val["condition"] == cond), "nDCG@10"].mean())


def select_best(val: pd.DataFrame, names: list[str]) -> str:
    """Condition of ``names`` with the highest mean VAL exact nDCG@10 (the first on ties)."""
    best, best_m = names[0], -1.0
    for c in names:
        m = exact_ndcg(val, c)
        if m > best_m:
            best, best_m = c, m
    return best


def select_lambda(val: pd.DataFrame, cond: str) -> float:
    """Lambda with the highest mean VAL exact nDCG@10 for ``cond`` (the smaller on ties);
    ``val`` holds conditions ``cond|label`` for every lambda of LAMBDAS."""
    best, best_m = None, -1.0
    for lam in LAMBDAS:
        m = exact_ndcg(val, f"{cond}|{lam_label(lam)}")
        if m > best_m:
            best, best_m = lam, m
    return best


def fuse(s: np.ndarray, aux: np.ndarray, lam: float) -> np.ndarray:
    """``s + lam * aux``; for lam = inf, ``INF_SCALE * aux + s`` in float64 (``aux`` orders
    first, ``s`` only breaks the ties of ``aux``)."""
    if math.isinf(lam):
        return INF_SCALE * aux.astype(np.float64) + s
    return s + s.dtype.type(lam) * aux.astype(s.dtype, copy=False)


def lambda_tops(
    s: np.ndarray,
    aux: np.ndarray,
    tie_rank: np.ndarray,
    k: int,
    lambdas=LAMBDAS,
    plain_at_zero: bool = False,
) -> dict[float, np.ndarray]:
    """Top-``k`` items (ties by ``tie_rank``) of ``fuse(s, aux, lam)`` for every lambda.
    With ``plain_at_zero`` the lambda = 0 list is ``ranking.top_k_scores(s, k)``, the ranking
    of run_corrections (for a content score ``s`` whose lists must match the grid)."""
    return {
        lam: top_k_scores(s, k)
        if (plain_at_zero and lam == 0)
        else top_k_scores_ties(fuse(s, aux, lam), k, tie_rank)
        for lam in lambdas
    }


def check_against_grid(
    df: pd.DataFrame, stem: str, split: str, equiv: dict[str, str], tol: float = 1e-4
) -> dict[str, float]:
    """Largest per-user difference to the file of run_corrections for the conditions
    ``equiv`` (new name -> grid name) and for the baselines, at every level the two share."""
    grid = pd.read_csv(
        OUT_DIR / f"{stem}.csv.gz", usecols=["user_id", "split", "gt", "condition", *METRICS]
    )
    grid = grid[grid["split"] == split]
    names = {
        **equiv,
        "popularity": "popularity",
        "random": "random",
        "top_category": "top_category",
    }
    diffs = {}
    for gt in sorted(set(df["gt"]) & set(grid["gt"])):
        for new, old in names.items():
            a = df[(df["gt"] == gt) & (df["condition"] == new)].set_index("user_id")
            if a.empty:
                continue
            b = grid[(grid["gt"] == gt) & (grid["condition"] == old)].set_index("user_id")
            assert set(a.index) == set(b.index), f"{split} {gt}: users differ from the grid run"
            d = float((a[METRICS] - b.loc[a.index, METRICS]).abs().to_numpy().max())
            assert d < tol, f"{split} {gt} {new}: differs from run_corrections by {d}"
            diffs[f"{split}_{gt}_{new}"] = d
    return diffs


def run_info(t0: float) -> dict:
    return {
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }


def write_run(stem: str, per_user: pd.DataFrame, summary: dict) -> None:
    """Per-user TEST metrics (``.csv.gz``, git-ignored) and the JSON summary."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    per_user.to_csv(OUT_DIR / f"{stem}.csv.gz", index=False, float_format="%.6g")
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    log.info(
        f"saved {OUT_DIR / stem}.[csv.gz|json]  runtime {summary['runtime_s']}s  "
        f"peak RSS {summary['peak_rss_mb']} MB"
    )


def provenance_of(pop: Population, *extra: Path) -> dict:
    return provenance([*pop.inputs, *extra])


# ─────────────────────────────────────────────────────────────────────────────
# Paired statistics
# ─────────────────────────────────────────────────────────────────────────────


def wide_test(stem: str, gt: str = "exact") -> dict[str, pd.DataFrame]:
    """metric short name -> users x conditions table of the TEST rows of ``stem``'s per-user file."""
    df = pd.read_csv(
        OUT_DIR / f"{stem}.csv.gz", usecols=["user_id", "gt", "condition", *STAT_METRICS]
    )
    df = df[df["gt"] == gt]
    out = {}
    for metric, short in STAT_METRICS.items():
        w = df.pivot(index="user_id", columns="condition", values=metric).sort_index()
        out[short] = w
    return out


def paired_rows(
    wide: pd.DataFrame, pairs: list[tuple[str, str, str]], meta: dict, family: str
) -> list[dict]:
    """Paired tests of ``(test id, condition, reference)`` triples on ``wide`` (users x
    conditions), Holm-adjusted together as one family."""
    rows = []
    for tid, cond, ref in pairs:
        r = paired_row(wide[cond].to_numpy(), wide[ref].to_numpy(), N_BOOT, N_FLIP)
        rows.append(
            {**meta, "family": family, "test": tid, "condition": cond, "reference": ref, **r}
        )
    add_holm_columns(rows)
    return rows


def verdict(row: dict) -> str:
    """'better', 'worse' or 'ns' by the primary test (Holm-adjusted sign-flip p, alpha 0.05)."""
    if not row["p_holm_signflip"] < ALPHA or row["mean_diff"] == 0:  # also NaN
        return "ns"
    return "better" if row["mean_diff"] > 0 else "worse"


def tally(rows: list[dict], by: list[str]) -> pd.DataFrame:
    """Counts of better / worse / ns verdicts of ``rows`` per group of the columns ``by``,
    separately for verified and unverified encoders."""
    if not rows:
        return pd.DataFrame(columns=[*by, "encoders", "n_tests", "better", "worse", "ns"])
    df = pd.DataFrame(rows)
    df["verdict"] = [verdict(r) for r in rows]
    df["encoders"] = np.where(df["encoder"].isin(UNVERIFIED_ENCODERS), "unverified", "verified")
    out = df.groupby([*by, "encoders", "verdict"]).size().unstack("verdict", fill_value=0)
    for c in ("better", "worse", "ns"):
        if c not in out:
            out[c] = 0
    out["n_tests"] = out[["better", "worse", "ns"]].sum(axis=1)
    return out[["n_tests", "better", "worse", "ns"]].reset_index()


def flat_rows(summary: dict) -> list[dict]:
    """One row per condition of a run summary (``val`` and ``test`` blocks of
    ``block_means``): VAL exact nDCG@10, TEST exact nDCG@10 / Recall@100 / users with
    a hit, TEST depth-2 nDCG@10 and the cold / warm nDCG@10, Recall@100 and user counts."""
    test, val = summary["test"], summary.get("val", {}).get("exact", {})
    rows = []
    for cond, m in test["exact"].items():
        row = {
            "domain": summary["domain"],
            "encoder": summary.get("encoder", ""),
            "text": summary.get("text", "titlecat"),
            "condition": cond,
            "val_ndcg10": val[cond]["nDCG@10"] if cond in val else float("nan"),
            "ndcg10": m["nDCG@10"],
            "r100": m["R@100"],
            "n_users": m["n"],
            "n_hit10": m["n_hit10"],
            "n_hit100": m["n_hit100"],
        }
        if "d2" in test:
            row["d2_ndcg10"] = test["d2"][cond]["nDCG@10"]
        for g in ("cold", "warm"):
            if g in test and cond in test[g]:
                row[f"{g}_ndcg10"] = test[g][cond]["nDCG@10"]
                row[f"{g}_r100"] = test[g][cond]["R@100"]
                row[f"{g}_n_users"] = test[g][cond]["n"]
                row[f"{g}_n_hit10"] = test[g][cond]["n_hit10"]
        rows.append(row)
    return rows
