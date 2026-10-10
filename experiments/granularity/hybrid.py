"""
Does content add anything over the train-based popularity list at item level?

Hybrid score of a (user, item) pair, for a content condition C of
experiments.granularity.run_corrections:

    score(u, i) = cos_C(profile_u, item_i) + lambda * p(i)

where p is the z-score over the catalog of log(1 + n_i) and n_i is the number
of TRAIN users whose exact target ground truth contains the item
(acw.evaluation.pipeline.train_popularity). Conditions C: ``raw``,
``zca_profile`` and ``zca_catalog_a1.0``. lambda = 0 is the pure content
condition and lambda = inf is the popularity ranking with exact ties broken by
cosine. The full catalog is ranked (top 100 per user, users in batches).

lambda is chosen per (domain, C) on the VAL users by mean exact nDCG@10 (ties:
the smaller lambda); TEST metrics (exact nDCG@10 and Recall@{10,50,100}, depth-2
category nDCG@10) are reported for the chosen lambda. The users, profiles,
fitted moments and constant baselines are those of run_corrections, and the
pure conditions and the popularity list are checked against its per-user output.

Outputs (results/granularity/):
  hybrid_{domain}_{encoder}.csv.gz  per-user TEST metrics of the baselines, the
                                    pure conditions and the selected hybrids
  hybrid_{domain}_{encoder}.json    lambda grid, VAL curve (selected on it), TEST
                                    curve (diagnostic only), selected lambda,
                                    TEST means, grid check, provenance
  stats_hybrid.csv                  (--stats) paired tests on TEST, Holm within
                                    (domain, metric); needs the three domains

Usage:
    python -m experiments.granularity.hybrid --domain music --encoder gte
    python -m experiments.granularity.hybrid --stats --encoder gte
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import resource
import time

import numpy as np
import pandas as pd

from acw.corrections import fit_moments, zca_power
from acw.data.splits import SPLIT_NAMES, load_split, split_path
from acw.embeddings.generate_h2_profiles.io import load_user_history, user_history_path
from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_ground_truth, item_genres
from acw.evaluation.pipeline import MAX_K, train_popularity, trivial_rankings
from acw.evaluation.ranking import top_k_scores
from acw.evaluation.significance import add_holm_columns, holm, paired_row
from acw.provenance import provenance
from acw.settings import GRANULARITY_DIR, PROCESSED_DIR
from experiments.granularity.run_corrections import (
    METRICS,
    REGISTRY,
    _load_matrix,
    h2_profiles,
    score_split,
)

log = logging.getLogger("hybrid")

OUT_DIR = GRANULARITY_DIR
TEXT = "titlecat"
DOMAINS = ("music", "movies", "games")
CONDITIONS = ("raw", "zca_profile", "zca_catalog_a1.0")
LAMBDAS = (0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, math.inf)
INF_SCALE = 1e6  # lambda = inf: popularity z-score * INF_SCALE + cosine (float64)
BATCH = 512  # users per score matrix
GRID_TOL = 1e-4  # per-user agreement with run_corrections (its CSV keeps 6 significant digits)
STAT_METRICS = {"nDCG@10": "ndcg10", "R@100": "r100"}
# (condition, reference) of the paired tests of --stats
PAIRS = (
    *((f"hybrid_{c}", "popularity") for c in CONDITIONS),
    *((f"hybrid_{c}", c) for c in CONDITIONS),
    ("hybrid_zca_profile", "hybrid_raw"),
    ("hybrid_zca_profile", "hybrid_zca_catalog_a1.0"),
)
N_BOOT = 2_000
N_FLIP = 10_000


def lam_label(lam: float) -> str:
    return "inf" if math.isinf(lam) else f"{lam:g}"


def hybrid_top(
    q: np.ndarray, x: np.ndarray, pz: np.ndarray, lambdas=LAMBDAS, k: int = MAX_K
) -> dict[float, np.ndarray]:
    """Top-``k`` item indices per query for every lambda: ``q @ x.T + lambda * pz``
    over the whole catalog, one matrix product per batch of ``BATCH`` queries."""
    pz64 = pz.astype(np.float64)
    out: dict[float, list[np.ndarray]] = {lam: [] for lam in lambdas}
    for i in range(0, len(q), BATCH):
        s = q[i : i + BATCH] @ x.T
        for lam in lambdas:
            # lambda = inf: float64 so that the cosine only orders items with equal counts
            sc = INF_SCALE * pz64 + s if math.isinf(lam) else s + np.float32(lam) * pz
            out[lam].append(top_k_scores(sc, k))
    return {lam: np.vstack(v) for lam, v in out.items()}


def popularity_z(counts: np.ndarray) -> np.ndarray:
    """z-score over the catalog of log(1 + count), float32."""
    lp = np.log1p(counts.astype(np.float64))
    pz = (lp - lp.mean()) / lp.std()
    gaps = np.diff(np.unique(pz))
    assert INF_SCALE * gaps.min() > 4.0, "INF_SCALE too small to keep the count order"
    return pz.astype(np.float32)


def exact(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["gt"] == "exact"]


def select_lambda(val: pd.DataFrame, cond: str) -> float:
    """Lambda with the highest mean VAL exact nDCG@10 for condition ``cond``
    (the smaller lambda on ties); ``val`` has conditions ``cond|label``."""
    best, best_m = None, -1.0
    for lam in LAMBDAS:
        m = val.loc[val["condition"] == f"{cond}|{lam_label(lam)}", "nDCG@10"].mean()
        if m > best_m:
            best, best_m = lam, m
    return best


def pure_and_popularity(df: pd.DataFrame) -> pd.DataFrame:
    """Rows of the pure conditions (lambda = 0, renamed ``C``) and of popularity."""
    out = df[df["condition"].isin([f"{c}|0" for c in CONDITIONS] + ["popularity"])].copy()
    out["condition"] = out["condition"].str.removesuffix("|0")
    return out


def check_against_grid(df: pd.DataFrame, stem: str, split: str) -> dict[str, float]:
    """Max absolute per-user difference to the per-user file of run_corrections for the
    conditions both runs compute (pure conditions, popularity), at every level of ``df``."""
    grid = pd.read_csv(
        OUT_DIR / f"{stem}.csv.gz", usecols=["user_id", "split", "gt", "condition", *METRICS]
    )
    grid = grid[grid["split"] == split]
    diffs = {}
    for gt in df["gt"].unique():
        for cond in (*CONDITIONS, "popularity"):
            a = df[(df["gt"] == gt) & (df["condition"] == cond)].set_index("user_id")
            b = grid[(grid["gt"] == gt) & (grid["condition"] == cond)].set_index("user_id")
            assert set(a.index) == set(b.index), f"{split} {gt}: users differ from the grid run"
            d = float((a[METRICS] - b.loc[a.index, METRICS]).abs().to_numpy().max())
            assert d < GRID_TOL, f"{split} {gt} {cond}: differs from run_corrections by {d}"
            diffs[f"{split}_{gt}_{cond}"] = d
    return diffs


def run(domain: str, encoder: str) -> None:
    t0 = time.time()
    key = (domain, encoder, TEXT)
    src = REGISTRY[key]
    cfg = resolve_paths(domain)
    items, item_idx = _load_matrix(src.items, src.items_idx)
    books, book_idx = _load_matrix(src.books, src.books_idx)
    with open(cfg.items_idx_path) as f:
        if set(json.load(f)) != set(item_idx):
            raise ValueError("encoder item ids differ from the domain catalog")
    idx_to_item = {v: k for k, v in item_idx.items()}

    # Population of run_corrections: an H2 profile and a categorised exact item
    history = load_user_history(domain)
    train_users = load_split(domain, "train")
    exact_gt = build_ground_truth(cfg, "exact")
    categorised = set(item_genres(cfg, 1))
    train_prof, _, _ = h2_profiles(train_users, history, books, book_idx)
    eval_prof, eval_users = {}, {}
    for s in ("val", "test"):
        prof, kept, _ = h2_profiles(load_split(domain, s), history, books, book_idx)
        ok = [i for i, u in enumerate(kept) if exact_gt.get(u, set()) & categorised]
        eval_prof[s], eval_users[s] = prof[ok], [kept[i] for i in ok]
    del books
    log.info(
        f"{domain}: items {items.shape}, eval users { {s: len(u) for s, u in eval_users.items()} }"
    )

    # Popularity feature: train users only
    pop = train_popularity(cfg, train_users)
    pz = popularity_z(np.array([pop.get(idx_to_item[i], 0) for i in range(len(idx_to_item))]))

    mom_c, mom_p = fit_moments(items), fit_moments(train_prof)
    zca_items = zca_power(items, *mom_c, 1.0)
    pairs = {  # condition -> (profile transform, items)
        "raw": (lambda q: q, items),
        "zca_profile": (lambda q: zca_power(q, *mom_p, 1.0), items),
        "zca_catalog_a1.0": (lambda q: q, zca_items),
    }
    rankings: dict[str, dict[str, np.ndarray]] = {s: {} for s in eval_prof}
    for cond, (qfn, x) in pairs.items():
        for s, q in eval_prof.items():
            for lam, top in hybrid_top(qfn(q), x, pz).items():
                rankings[s][f"{cond}|{lam_label(lam)}"] = top
        log.info(f"ranked {cond}")

    by_depth = {
        d: trivial_rankings(cfg, idx_to_item, train_users, depth=d)[0]
        for d in {cfg.category_depth, 2}
    }
    constants = {"exact": by_depth[cfg.category_depth], "d2": by_depth[2]}
    args = (None, exact_gt, cfg, idx_to_item, True)
    val, *_ = score_split("val", eval_users["val"], rankings["val"], constants, {}, *args)
    test, *_ = score_split(
        "test", eval_users["test"], rankings["test"], constants, {"d2": 2}, *args
    )
    val, test = pd.concat(val, ignore_index=True), pd.concat(test, ignore_index=True)
    log.info("scored val and test")

    # Pure conditions and popularity must reproduce run_corrections
    stem = f"{domain}_{encoder}_{TEXT}"
    grid_check = {
        **check_against_grid(pure_and_popularity(val), stem, "val"),
        **check_against_grid(pure_and_popularity(test), stem, "test"),
    }

    # Select lambda on VAL (exact nDCG@10) and report TEST
    chosen = {c: select_lambda(exact(val), c) for c in CONDITIONS}
    log.info(f"selected lambda: { {c: lam_label(v) for c, v in chosen.items()} }")
    keep = {"random": "random", "popularity": "popularity", "top_category": "top_category"}
    for c in CONDITIONS:
        keep[f"{c}|0"] = c
        keep[f"{c}|{lam_label(chosen[c])}"] = f"hybrid_{c}"
    out = test[test["condition"].isin(keep)].copy()
    out["condition"] = out["condition"].map(keep)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem_out = f"hybrid_{domain}_{encoder}"
    out.to_csv(OUT_DIR / f"{stem_out}.csv.gz", index=False, float_format="%.6g")

    def means(df: pd.DataFrame) -> dict:
        g = df.groupby(["gt", "condition"], sort=False)
        m, n = g[METRICS].mean(), g.size()
        return {
            gt: {c: {**{k: float(m.loc[(gt, c), k]) for k in METRICS}, "n": int(n.loc[(gt, c)])}
                 for c in df.loc[df["gt"] == gt, "condition"].unique()}
            for gt in df["gt"].unique()
        }  # fmt: skip

    def curve(df: pd.DataFrame) -> dict:
        m = means(exact(df))["exact"]
        return {
            c: {lam_label(lam): m[f"{c}|{lam_label(lam)}"] for lam in LAMBDAS} for c in CONDITIONS
        }

    inputs = [PROCESSED_DIR / f for f in (src.items, src.items_idx, src.books, src.books_idx)]
    inputs += [cfg.items_idx_path, cfg.items_meta_path, cfg.gt_path, user_history_path(domain)]
    inputs += [split_path(domain, s) for s in SPLIT_NAMES]
    summary = {
        "domain": domain,
        "encoder": encoder,
        "model": src.model,
        "text": TEXT,
        "score": "cosine_C(profile, item) + lambda * z(log(1 + n_train_users(item)))",
        "conditions": list(CONDITIONS),
        "lambdas": [lam_label(lam) for lam in LAMBDAS],
        "selection": "VAL users, mean exact nDCG@10, ties to the smaller lambda",
        "n_eval_users": {s: len(u) for s, u in eval_users.items()},
        "selected_lambda": {c: lam_label(v) for c, v in chosen.items()},
        "val_curve": curve(val),
        "test_curve_diagnostic": curve(test),
        "test": means(out),
        "grid_check_max_abs_diff": grid_check,
        "provenance": provenance(inputs),
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    (OUT_DIR / f"{stem_out}.json").write_text(json.dumps(summary, indent=2))
    log.info(f"saved {OUT_DIR / stem_out}.[csv.gz|json]  runtime {summary['runtime_s']}s")


def stats(encoder: str) -> None:
    """Paired tests on TEST (exact nDCG@10 and Recall@100) of ``PAIRS``; Holm within
    each (domain, metric) family of ``len(PAIRS)`` tests. Columns as in stats.py: the
    PRIMARY sign-flip test on the mean (``p_signflip``, ``p_holm_signflip``), the secondary
    Wilcoxon-Pratt test (``p_wilcoxon``, ``p_holm_wilcoxon``; ``p_holm`` is the same
    Holm-adjusted value, kept for the tables that use it) and ``n_nonzero_diff``."""
    rows, files = [], []
    for domain in DOMAINS:
        path = OUT_DIR / f"hybrid_{domain}_{encoder}.csv.gz"
        files.append(path)
        df = pd.read_csv(path, usecols=["user_id", "gt", "condition", *STAT_METRICS])
        df = df[df["gt"] == "exact"]
        for metric, short in STAT_METRICS.items():
            wide = df.pivot(index="user_id", columns="condition", values=metric).sort_index()
            assert not wide.isna().any().any(), f"{domain}: users missing conditions"
            fam = [
                {"domain": domain, "encoder": encoder, "metric": short, "condition": cond,
                 "reference": ref,
                 **paired_row(wide[cond].to_numpy(), wide[ref].to_numpy(), N_BOOT, N_FLIP)}
                for cond, ref in PAIRS
            ]  # fmt: skip
            p_adj, _ = holm([r["p_wilcoxon"] for r in fam])
            add_holm_columns(fam)
            rows += [{**r, "p_holm": float(p)} for r, p in zip(fam, p_adj)]
    pd.DataFrame(rows).to_csv(OUT_DIR / "stats_hybrid.csv", index=False, float_format="%.6g")
    prov = {"encoder": encoder, "pairs": PAIRS, "provenance": provenance(files)}
    (OUT_DIR / "stats_hybrid_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info(f"saved {OUT_DIR / 'stats_hybrid.csv'} ({len(rows)} rows)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=DOMAINS)
    ap.add_argument("--encoder", default="gte")
    ap.add_argument("--stats", action="store_true", help="paired tests over the three domains")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.stats:
        stats(args.encoder)
    elif args.domain:
        run(args.domain, args.encoder)
    else:
        ap.error("give --domain or --stats")


if __name__ == "__main__":
    main()
