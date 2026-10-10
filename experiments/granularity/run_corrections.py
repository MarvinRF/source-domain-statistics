"""
Training-free geometric corrections of frozen text embeddings for zero-shot
book -> {music, movies, games} recommendation, under a ladder of relevance
granularities.

For one (domain, encoder, text variant) the script builds H2 profiles (mean of
the user's last MAX_BOOKS_PROFILE book embeddings, L2-normalized), applies each
correction of acw.corrections, ranks the target catalog by cosine similarity
and scores per-user nDCG@10 / Recall@{10,50,100} on the val and test users of
the disjoint splits in data/splits/ (python -m acw.data.splits). Everything is
fitted on the catalog or on TRAIN-user profiles only.

Relevance granularities (column ``gt`` of the output): ``exact`` (the items the
user interacted with) and ``d3``/``d2``/``d1`` (all catalog items sharing the
category at that index of the item ``categories`` list, clamped to the deepest
level an item has). A depth is skipped for a domain when fewer than
MIN_DEPTH_COVERAGE of its catalog items have a category at that level. All
granularities are evaluated on the SAME users: those with an H2 profile and at
least one categorised exact item (no cap on the number of categories per user).
The ``top_category`` baseline is recomputed per depth (the category reached by
most train users; ``exact`` uses depth 2).

Conditions (profile / items):
  raw                      raw / raw
  center_catalog           raw / centered with the catalog mean
  abtt_catalog_D{1,3,10}   raw / All-but-the-Top on the catalog, D directions
  pca_catalog              raw / PCA-whitened with the catalog moments
  zca_catalog_a{0.25,0.5,0.75,1.0}
                           raw / partial ZCA (alpha=1 is the paper's method)
  zca_profile              ZCA fitted on train-user profiles / raw
  zca_both_separate        ZCA of train-user profiles / ZCA of the catalog
  zca_symmetric            catalog ZCA / catalog ZCA
  zca_catalog_books        raw / ZCA fitted on the book catalog
  zca_catalog_grocery      raw / ZCA fitted on the grocery catalog (gte,
                           titlecat only; skipped and logged otherwise)
  random, popularity, top_category
                           non-personalised baselines, built from train users
                           only: popularity = number of train users whose exact
                           ground truth contains the item; top_category = the
                           category reached by most train users, its items in
                           popularity order

Inputs: embeddings listed in REGISTRY (data/processed/...), data/raw user
histories and ground truth, data/splits/{domain}_{train,val,test}.json (the split
of ACW_SPLIT_SEED, default 42; another seed S reads data/splits/seed{S}/ and writes
results/granularity/seed{S}/).
Outputs: results/granularity/{domain}_{encoder}_{text}.csv.gz (per-user
metrics) and .json (means, counts, category-depth statistics, files, runtime,
peak RSS). With --legacy-population the canonical val users of the older
protocol (music only; train = canonical train; depth 2 only; users spanning
more than MAX_GENRES categories excluded) are evaluated instead, with the
metadata ``rating_number`` as item popularity (as in that protocol), and the
outputs get a _legacy suffix.

Usage:
    python -m experiments.granularity.run_corrections --domain music --encoder gte --text titlecat
    python -m experiments.granularity.run_corrections --domain music --encoder gte --text desc
    python -m experiments.granularity.run_corrections --domain music --encoder gte --text title
    python -m experiments.granularity.run_corrections --domain music --encoder gte --text titlecat --legacy-population
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time
from collections import defaultdict
from dataclasses import dataclass
from functools import partial

import numpy as np
import pandas as pd

from acw.corrections import abtt, center, fit_moments, l2_normalize, pca_whiten, zca_power
from acw.data.splits import SPLIT_NAMES, load_split, split_path
from acw.embeddings.generate_h2_profiles.io import load_user_history, user_history_path
from acw.evaluation.config import MAX_GENRES, resolve_paths
from acw.evaluation.loader import build_ground_truth, item_genres
from acw.evaluation.pipeline import (
    MAX_K,
    RECALL_K,
    _score,
    compute_random_baseline,
    trivial_rankings,
)
from acw.evaluation.ranking import top_k
from acw.provenance import provenance
from acw.settings import GRANULARITY_DIR, MAX_BOOKS_PROFILE, PROCESSED_DIR, SEED

log = logging.getLogger("run_corrections")

OUT_DIR = GRANULARITY_DIR  # results/granularity/, or seed{S}/ under ACW_SPLIT_SEED=S
METRICS = ["nDCG@10", *(f"R@{k}" for k in RECALL_K)]
DEPTHS = (3, 2, 1)  # category depths of the granularity ladder, finest first
MIN_DEPTH_COVERAGE = 0.5  # skip a depth when fewer catalog items have a category at it
GENRE_CHUNK = 250  # users per category-GT build; bounds the memory of the expanded sets
ALPHAS = (0.25, 0.5, 0.75, 1.0)
ABTT_D = (1, 3, 10)

# ─────────────────────────────────────────────────────────────────────────────
# Encoder registry (every file verified on disk; see _load_matrix)
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Source:
    model: str
    items: str  # paths relative to PROCESSED_DIR
    items_idx: str
    books: str
    books_idx: str
    grocery: str | None = None
    grocery_idx: str | None = None
    provenance_unverified: bool = False  # row alignment checked, creation not documented


_GROCERY = ("grocery/grocery_embeddings_qwen.npy", "grocery/grocery_id_to_idx_qwen.json")
_BS = "backbone_screening"
_SCREENING = {  # file tag -> model
    "bge-m3": "BAAI/bge-m3",
    "e5-large-v2": "intfloat/e5-large-v2",
    "minilm": "sentence-transformers/all-MiniLM-L6-v2",
    "nomic": "nomic-ai/nomic-embed-text-v1.5",
    "snowflake-arctic-embed-l": "Snowflake/snowflake-arctic-embed-l",
    "uae-large-v1": "WhereIsAI/UAE-Large-V1",
}
_GTE = "Alibaba-NLP/gte-large-en-v1.5"
_BGE = "BAAI/bge-large-en-v1.5"
_QWEN3 = "Qwen/Qwen3-Embedding-8B"
_QWEN3_CKPT = "{d}/{d}_embeddings_checkpoint__home_recsys_models_qwen3-embedding-8b.npy"

REGISTRY: dict[tuple[str, str, str], Source] = {}
_BOOKS_GTE = ("book_embeddings.npy", "book_id_to_idx.json")

REGISTRY[("music", "gte", "titlecat")] = Source(
    _GTE, "album_embeddings.npy", "album_id_to_idx.json", *_BOOKS_GTE, *_GROCERY
)
REGISTRY[("music", "gte", "desc")] = Source(
    _GTE,
    "desc/albums_embeddings_qwen.npy",
    "desc/albums_id_to_idx_qwen.json",
    "desc/books_embeddings_qwen.npy",
    "desc/books_id_to_idx_qwen.json",
)
# title: the same albums and books embedded from the title only (no categories)
REGISTRY[("music", "gte", "title")] = Source(
    _GTE,
    "title/albums_embeddings_qwen.npy",
    "title/albums_id_to_idx_qwen.json",
    "title/books_embeddings_qwen.npy",
    "title/books_id_to_idx_qwen.json",
)
# bge-large: the bge_revalidation set (all four catalogs from one script, the
# one behind the paper's BGE analysis). The older music files in data/processed/
# (album_embeddings_bge-large.npy, *_checkpoint_BAAI_*) are different embeddings
# of the same model (row-wise cosine 0.78-0.94) and are not used.
REGISTRY[("music", "bge-large", "titlecat")] = Source(
    _BGE,
    "bge_revalidation/albums_embeddings_bge.npy",
    "bge_revalidation/albums_id_to_idx_bge.json",
    "bge_revalidation/books_embeddings_bge.npy",
    "bge_revalidation/books_id_to_idx_bge.json",
)
REGISTRY[("music", "qwen3", "titlecat")] = Source(
    _QWEN3, "album_embeddings_qwen3.npy", "album_id_to_idx.json",
    "book_embeddings_qwen3.npy", "book_id_to_idx.json",
)  # fmt: skip
REGISTRY[("music", "minilm-l12", "titlecat")] = Source(
    "sentence-transformers/all-MiniLM-L12-v2",
    "album_embeddings_checkpoint_all-MiniLM-L12-v2.npy", "album_id_to_idx.json",
    "book_embeddings_checkpoint_all-MiniLM-L12-v2.npy", "book_id_to_idx.json",
)  # fmt: skip
for _tag, _model in _SCREENING.items():
    REGISTRY[("music", _tag, "titlecat")] = Source(
        _model,
        f"{_BS}/albums_embeddings_{_tag}.npy", f"{_BS}/albums_id_to_idx_{_tag}.json",
        f"{_BS}/books_embeddings_{_tag}.npy", f"{_BS}/books_id_to_idx_{_tag}.json",
    )  # fmt: skip
for _d in ("movies", "games"):
    REGISTRY[(_d, "gte", "titlecat")] = Source(
        _GTE, f"{_d}/{_d}_embeddings_qwen.npy", f"{_d}/{_d}_id_to_idx_qwen.json",
        *_BOOKS_GTE, *_GROCERY,
    )  # fmt: skip
    REGISTRY[(_d, "bge-large", "titlecat")] = Source(
        _BGE, f"bge_revalidation/{_d}_embeddings_bge.npy",
        f"bge_revalidation/{_d}_id_to_idx_bge.json",
        "bge_revalidation/books_embeddings_bge.npy", "bge_revalidation/books_id_to_idx_bge.json",
    )  # fmt: skip
    REGISTRY[(_d, "qwen3", "titlecat")] = Source(
        _QWEN3, _QWEN3_CKPT.format(d=_d), f"{_d}/{_d}_id_to_idx_qwen.json",
        "book_embeddings_qwen3.npy", "book_id_to_idx.json",
        provenance_unverified=True,
    )  # fmt: skip


# Encoders whose embeddings failed or could not run the re-encoding audit of docs/EMBEDDINGS.md
UNVERIFIED_ENCODERS = frozenset({"qwen3", "minilm-l12"})


def _load_matrix(emb: str, idx: str) -> tuple[np.ndarray, dict[str, int]]:
    """L2-normalized float32 embeddings and their id -> row map; the row count
    must equal the id map size and the map must cover rows 0..n-1 exactly."""
    x = np.load(PROCESSED_DIR / emb)
    with open(PROCESSED_DIR / idx) as f:
        id_map = json.load(f)
    if x.shape[0] != len(id_map) or set(id_map.values()) != set(range(x.shape[0])):
        raise ValueError(f"{emb}: {x.shape[0]} rows vs {len(id_map)} ids in {idx}")
    x = x.astype(np.float32, copy=False)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
    return x, id_map


# ─────────────────────────────────────────────────────────────────────────────
# Profiles
# ─────────────────────────────────────────────────────────────────────────────


def window_books(hist: list[str], book_idx: dict[str, int]) -> list[str]:
    """The books of a user's history that enter the H2 profile: the last
    MAX_BOOKS_PROFILE entries that are embedded (also the lexical query window)."""
    return [a for a in hist[-MAX_BOOKS_PROFILE:] if a in book_idx]


def h2_profiles(
    users: list[str], history: dict[str, list[str]], books: np.ndarray, book_idx: dict[str, int]
) -> tuple[np.ndarray, list[str], float]:
    """H2: L2-normalized mean of the user's last MAX_BOOKS_PROFILE book
    embeddings (those that are embedded). Users without any are dropped.
    Returns (profiles, kept users, mean number of books pooled)."""
    prof, keep, n_pooled = [], [], []
    for u in users:
        rows = [book_idx[a] for a in window_books(history.get(u, []), book_idx)]
        if rows:
            prof.append(books[rows].mean(0))
            keep.append(u)
            n_pooled.append(len(rows))
    return l2_normalize(np.asarray(prof)), keep, float(np.mean(n_pooled))


# ─────────────────────────────────────────────────────────────────────────────
# Scoring
# ─────────────────────────────────────────────────────────────────────────────


def _row(ranked: list[str], relevant: set[str]) -> list[float]:
    m: dict[str, list[float]] = defaultdict(list)
    _score(ranked, relevant, m)
    return [m[k][0] for k in METRICS]


def category_levels(cfg, catalog: set[str]) -> dict[str, int]:
    """Number of category levels of each catalog item that has metadata."""
    n_levels: dict[str, int] = {}
    with open(cfg.items_meta_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("parent_asin") in catalog:
                n_levels[item["parent_asin"]] = len(item.get("categories", []))
    return n_levels


def depth_stats(cfg, catalog: set[str]) -> dict[int, dict]:
    """Per category depth: share of catalog items that have a category at that
    level (``coverage``; shallower items are clamped to their deepest level
    when a ground truth is built) and the number of distinct categories."""
    n_levels = category_levels(cfg, catalog)
    out = {}
    for d in DEPTHS:
        genres = item_genres(cfg, d)
        out[d] = {
            "coverage": sum(1 for a in catalog if n_levels.get(a, 0) > d) / len(catalog),
            "n_categories": len({genres[a] for a in catalog if a in genres}),
        }
    return out


def score_split(
    split: str,
    users: list[str],
    rankings: dict[str, np.ndarray],
    constants: dict[str, dict[str, list[str]]],
    depths: dict[str, int],
    max_genres: int | None,
    exact_gt: dict[str, set[str]],
    cfg,
    idx_to_item: dict[int, str],
    strict: bool,
    extra_gt: dict[str, dict[str, set[str]]] | None = None,
) -> tuple[list[pd.DataFrame], dict[str, float], dict[str, dict]]:
    """Per-user metrics of every condition and baseline for one split under
    every granularity (``exact`` plus the modes of ``depths``: mode -> depth).
    ``constants`` maps each mode to its non-personalised ranked lists. With
    ``strict`` every user must have a non-empty ground truth in every mode.
    ``extra_gt`` adds modes with a precomputed ground truth (mode -> user ->
    relevant items, e.g. the cold-item and warm-item subsets of the exact
    ground truth); users with an empty relevant set are skipped in those modes.
    Returns (frames, random-baseline check values, relevant-set size stats)."""
    extra_gt = extra_gt or {}
    frames: list[pd.DataFrame] = []
    n_items = len(idx_to_item)
    random_check: dict[str, float] = {}
    rel_stats: dict[str, dict] = {}
    for mode in ("exact", *depths, *extra_gt):
        rng = np.random.default_rng(SEED)  # same draw order as compute_random_baseline
        rows: list[tuple] = []
        rel_pct: list[float] = []
        for start in range(0, len(users), GENRE_CHUNK):
            chunk = users[start : start + GENRE_CHUNK]
            if mode == "exact":
                gt = exact_gt
            elif mode in extra_gt:
                gt = extra_gt[mode]
            else:
                gt = build_ground_truth(
                    cfg, "genre", set(chunk), depth=depths[mode], max_genres=max_genres
                )
            for j, u in enumerate(chunk):
                relevant = gt.get(u)
                if not relevant:
                    if strict and mode not in extra_gt:
                        raise RuntimeError(f"user {u} has an empty {mode} ground truth")
                    continue
                pos = start + j
                rel_pct.append(100.0 * len(relevant) / n_items)
                rand = [idx_to_item[i] for i in rng.choice(n_items, size=MAX_K, replace=False)]
                lists = {"random": rand, **constants[mode]}
                for cond, top in rankings.items():
                    lists[cond] = [idx_to_item[i] for i in top[pos]]
                for cond, ranked in lists.items():
                    rows.append((u, split, mode, cond, *_row(ranked, relevant)))
            del gt
        df = pd.DataFrame(rows, columns=["user_id", "split", "gt", "condition", *METRICS])
        frames.append(df)
        rel_stats[mode] = {
            "n_users": len(rel_pct),
            "median_rel_pct_of_catalog": float(np.median(rel_pct)) if rel_pct else None,
            "mean_rel_pct_of_catalog": float(np.mean(rel_pct)) if rel_pct else None,
        }
        if mode == "exact" and len(df):
            ev = list(df.loc[df["condition"] == "random", "user_id"])
            ref = compute_random_baseline(exact_gt, ev, idx_to_item, n_items)["nDCG@10"]
            mine = float(df.loc[df["condition"] == "random", "nDCG@10"].mean())
            assert abs(ref - mine) < 1e-9, (ref, mine)
            random_check[f"{split}_exact_random_nDCG@10"] = mine
    return frames, random_check, rel_stats


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=["music", "movies", "games"], required=True)
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--text", choices=["titlecat", "desc", "title"], default="titlecat")
    ap.add_argument("--legacy-population", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    t0 = time.time()

    key = (args.domain, args.encoder, args.text)
    if key not in REGISTRY:
        avail = sorted(f"{e}/{t}" for d, e, t in REGISTRY if d == args.domain)
        raise SystemExit(f"no registry entry for {key}; available for {args.domain}: {avail}")
    src = REGISTRY[key]
    cfg = resolve_paths(args.domain)

    items, item_idx = _load_matrix(src.items, src.items_idx)
    books, book_idx = _load_matrix(src.books, src.books_idx)
    with open(cfg.items_idx_path) as f:
        if set(json.load(f)) != set(item_idx):
            raise ValueError("encoder item ids differ from the domain catalog")
    with open(PROCESSED_DIR / "book_id_to_idx.json") as f:
        if set(json.load(f)) != set(book_idx):
            raise ValueError("encoder book ids differ from the canonical book catalog")
    idx_to_item = {v: k for k, v in item_idx.items()}
    log.info(f"items {items.shape}, books {books.shape} ({src.model})")

    # Users and H2 profiles
    history = load_user_history(args.domain)
    if args.legacy_population:
        if not cfg.canonical_val_path.exists():
            raise SystemExit(f"no legacy canonical split for {args.domain}")
        train_users = json.load(open(cfg.canonical_train_path))
        split_users = {"val": json.load(open(cfg.canonical_val_path))}
    else:
        train_users = load_split(args.domain, "train")
        split_users = {s: load_split(args.domain, s) for s in ("val", "test")}
    exact_gt = build_ground_truth(cfg, "exact")
    categorised = set(item_genres(cfg, 1))  # items with at least one category
    catalog = set(item_idx)
    d_stats = depth_stats(cfg, catalog)
    if args.legacy_population:
        depths = {"d2": 2}
        max_genres: int | None = MAX_GENRES
    else:
        depths = {f"d{d}": d for d in DEPTHS if d_stats[d]["coverage"] >= MIN_DEPTH_COVERAGE}
        max_genres = None
    for d, st in d_stats.items():
        st["evaluated"] = f"d{d}" in depths
    log.info(f"category depths {d_stats}")

    train_prof, train_kept, _ = h2_profiles(train_users, history, books, book_idx)
    eval_prof, eval_users, n_dropped, n_uncat, mean_pooled = {}, {}, {}, {}, {}
    for s, users in split_users.items():
        prof, kept, mean_pooled[s] = h2_profiles(users, history, books, book_idx)
        n_dropped[s] = len(users) - len(kept)
        if not args.legacy_population:
            # one population for every granularity: a categorised exact item is required
            ok = [i for i, u in enumerate(kept) if exact_gt.get(u, set()) & categorised]
            n_uncat[s] = len(kept) - len(ok)
            prof, kept = prof[ok], [kept[i] for i in ok]
        eval_prof[s], eval_users[s] = prof, kept
    log.info(
        f"train profiles {train_prof.shape}; eval users "
        + ", ".join(
            f"{s}={len(u)} (no profile {n_dropped[s]}, uncategorised {n_uncat.get(s, 0)})"
            for s, u in eval_users.items()
        )
    )

    # Fitted moments: catalog, train profiles, book catalog, grocery catalog
    mom_c = fit_moments(items)
    mom_p = fit_moments(train_prof)
    mom_b = fit_moments(books)
    del books
    mom_g, skipped = None, []
    if src.grocery and args.text == "titlecat":
        groc, _ = _load_matrix(src.grocery, src.grocery_idx)
        mom_g = fit_moments(groc)
        del groc
    else:
        skipped.append("zca_catalog_grocery")
        log.info("zca_catalog_grocery skipped: no grocery embeddings for this encoder/text")
    log.info("moments fitted")

    # Conditions: (name, profile transform, item transform)
    def ident(q):
        return q

    def prof_train_zca(q):
        return zca_power(q, *mom_p, 1.0)

    def prof_catalog_zca(q):
        return zca_power(q, *mom_c, 1.0)

    zca_items = zca_power(items, *mom_c, 1.0)  # shared by a1.0, both_separate, symmetric
    specs = [
        ("raw", ident, lambda: items),
        ("center_catalog", ident, lambda: center(items, mom_c.mean)),
        *[
            (f"abtt_catalog_D{D}", ident, partial(abtt, items, mom_c.mean, mom_c.eigvecs, D))
            for D in ABTT_D
        ],
        ("pca_catalog", ident, lambda: pca_whiten(items, *mom_c)),
        *[
            (f"zca_catalog_a{a}", ident, partial(zca_power, items, *mom_c, a))
            if a != 1.0
            else (f"zca_catalog_a{a}", ident, lambda: zca_items)
            for a in ALPHAS
        ],
        ("zca_profile", prof_train_zca, lambda: items),
        ("zca_both_separate", prof_train_zca, lambda: zca_items),
        ("zca_symmetric", prof_catalog_zca, lambda: zca_items),
        ("zca_catalog_books", ident, lambda: zca_power(items, *mom_b, 1.0)),
    ]
    if mom_g is not None:
        specs.append(("zca_catalog_grocery", ident, lambda: zca_power(items, *mom_g, 1.0)))

    rankings: dict[str, dict[str, np.ndarray]] = {}
    for name, qfn, xfn in specs:
        x = xfn()
        rankings[name] = {s: top_k(qfn(q), x, MAX_K) for s, q in eval_prof.items()}
        del x
        log.info(f"ranked {name}")
    zca_items = None  # release the shared whitened catalog

    # Non-personalised baselines; top category per depth, chosen from train users only
    top_cat: dict[str, str | None] = {}
    by_depth: dict[int, dict[str, list[str]]] = {}
    for d in sorted({*depths.values(), cfg.category_depth}):
        by_depth[d], top_cat[f"d{d}"] = trivial_rankings(
            cfg, idx_to_item, train_users, depth=d, legacy_popularity=args.legacy_population
        )
    constants = {"exact": by_depth[cfg.category_depth]}
    constants.update({mode: by_depth[d] for mode, d in depths.items()})
    log.info(f"top category per depth: {top_cat}")

    frames, checks, rel_stats = [], {}, {}
    for s, users in eval_users.items():
        f, c, rel_stats[s] = score_split(
            s,
            users,
            {k: v[s] for k, v in rankings.items()},
            constants,
            depths,
            max_genres,
            exact_gt,
            cfg,
            idx_to_item,
            strict=not args.legacy_population,
        )
        frames += f
        checks.update(c)
        log.info(f"scored split {s}")
    df = pd.concat(frames, ignore_index=True)

    # Outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"{args.domain}_{args.encoder}_{args.text}" + (
        "_legacy" if args.legacy_population else ""
    )
    df.to_csv(OUT_DIR / f"{stem}.csv.gz", index=False, float_format="%.6g")
    grp = df.groupby(["split", "gt", "condition"], sort=False)
    means = grp[METRICS].mean()
    counts = grp.size()
    inputs = [
        PROCESSED_DIR / f
        for f in (src.items, src.items_idx, src.books, src.books_idx, "book_id_to_idx.json")
    ]
    if mom_g is not None:
        inputs += [PROCESSED_DIR / src.grocery, PROCESSED_DIR / src.grocery_idx]
    inputs += [cfg.items_idx_path, cfg.items_meta_path, cfg.gt_path, user_history_path(args.domain)]
    inputs += (
        [cfg.canonical_train_path, cfg.canonical_val_path]
        if args.legacy_population
        else [split_path(args.domain, s) for s in SPLIT_NAMES]
    )
    summary = {
        "domain": args.domain,
        "encoder": args.encoder,
        "text": args.text,
        "model": src.model,
        "population": "legacy_canonical_val" if args.legacy_population else "splits_70_10_20",
        "files": {
            k: v
            for k, v in {
                "items": src.items,
                "items_idx": src.items_idx,
                "books": src.books,
                "books_idx": src.books_idx,
                "grocery": src.grocery if mom_g is not None else None,
            }.items()
            if v
        },  # fmt: skip
        "n_train_profiles": len(train_kept),
        "n_eval_users": {s: len(u) for s, u in eval_users.items()},
        "provenance_unverified": src.provenance_unverified,
        "n_dropped_no_profile": n_dropped,
        "n_dropped_uncategorised": n_uncat,
        "mean_books_pooled_eval": mean_pooled,
        "granularities": ["exact", *depths],
        "category_depths": {f"d{d}": st for d, st in d_stats.items()},
        "relevance_size": rel_stats,
        "top_category": top_cat,
        "skipped_conditions": skipped,
        "random_baseline_check": checks,
        "means": {
            s: {
                g: {
                    c: {
                        **{m: float(means.loc[(s, g, c), m]) for m in METRICS},
                        "n": int(counts.loc[(s, g, c)]),
                    }
                    for c in df.loc[(df["split"] == s) & (df["gt"] == g), "condition"].unique()
                }
                for g in ("exact", *depths)
            }
            for s in eval_users
        },
        "provenance": provenance(inputs),
        "runtime_s": round(time.time() - t0, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
    }
    (OUT_DIR / f"{stem}.json").write_text(json.dumps(summary, indent=2))
    log.info(f"saved {OUT_DIR / stem}.[csv.gz|json]  runtime {summary['runtime_s']}s  "
             f"peak RSS {summary['peak_rss_mb']} MB")  # fmt: skip


if __name__ == "__main__":
    main()
