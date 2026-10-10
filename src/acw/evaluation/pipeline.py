"""Evaluation loop: random and non-personalised baselines, per-method scoring,
results table and CSV persistence. Driven by ``acw.evaluation.cli``.

Usage::

    python -m acw.evaluation.evaluate --all
"""

from __future__ import annotations

import csv
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from .config import (
    K_VALUES,
    KNOWN_METHODS,
    MAX_BOOKS,
    MAX_GENRES,
    N_TEST_USERS,
    PROCESSED,
    RESULTS_DIR,
    SEED,
    DomainConfig,
)
from .loader import item_genres, item_popularity, load_profiles
from .metrics import ndcg_at_k, precision_at_k, recall_at_k
from .ranking import top_k

log = logging.getLogger(__name__)

RECALL_K: list[int] = [10, 50, 100]
MAX_K: int = max(K_VALUES + RECALL_K)


def _score(ranked: list[str], relevant: set[str], metrics: dict[str, list[float]]) -> None:
    for k in K_VALUES:
        metrics[f"nDCG@{k}"].append(ndcg_at_k(ranked, relevant, k))
        metrics[f"P@{k}"].append(precision_at_k(ranked, relevant, k))
    for k in RECALL_K:
        metrics[f"R@{k}"].append(recall_at_k(ranked, relevant, k))


def compute_random_baseline(
    genre_gt: dict[str, set[str]],
    test_users: list[str],
    idx_to_item: dict[int, str],
    n_items: int,
) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    metrics: dict[str, list[float]] = defaultdict(list)

    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        idxs = rng.choice(n_items, size=MAX_K, replace=False)
        ranked = [idx_to_item[i] for i in idxs]
        _score(ranked, relevant, metrics)

    return {m: float(np.mean(v)) for m, v in metrics.items()}


def evaluate_method(
    label: str,
    profiles: np.ndarray,
    items_emb: np.ndarray,
    idx_map: dict[str, int],
    idx_to_item: dict[int, str],
    genre_gt: dict[str, set[str]],
    test_users: list[str],
) -> dict[str, float]:
    metrics: dict[str, list[float]] = defaultdict(list)
    users = [u for u in test_users if genre_gt.get(u) and idx_map.get(u) is not None]
    rows = np.array([idx_map[u] for u in users], dtype=np.int64)
    for uid, top in zip(users, top_k(profiles[rows], items_emb, MAX_K)):
        _score([idx_to_item[i] for i in top], genre_gt[uid], metrics)
    n_eval = len(users)

    avgs = {m: float(np.mean(v)) for m, v in metrics.items()}
    log.info(
        f"  {label:<50}: "
        f"nDCG@10={avgs.get('nDCG@10', 0):.4f}  "
        f"P@10={avgs.get('P@10', 0):.4f}  "
        f"(n={n_eval})"
    )
    avgs["_n_eval"] = n_eval
    return avgs


def evaluate_constant(
    ranked: list[str],
    gt: dict[str, set[str]],
    test_users: list[str],
) -> dict[str, float]:
    """Score the same ranked list for every user (non-personalised baselines)."""
    metrics: dict[str, list[float]] = defaultdict(list)
    for uid in test_users:
        if gt.get(uid):
            _score(ranked, gt[uid], metrics)
    return {m: float(np.mean(v)) for m, v in metrics.items()}


def train_category_counts(
    cfg: DomainConfig, train_users: list[str], depth: int | None = None
) -> dict[str, int]:
    """Number of ``train_users`` who consumed at least one item of each category
    (taxonomy ``depth``, default ``cfg.category_depth``)."""
    genres = item_genres(cfg, depth)
    with open(cfg.gt_path) as f:
        raw_gt: dict[str, list[str]] = json.load(f)
    counts: dict[str, int] = defaultdict(int)
    for uid in train_users:
        for g in {genres[a] for a in raw_gt.get(uid, []) if a in genres}:
            counts[g] += 1
    return dict(counts)


def train_popularity(cfg: DomainConfig, train_users: list[str]) -> dict[str, int]:
    """Number of ``train_users`` whose exact target ground truth contains each
    item. Unlike the dataset's ``rating_number`` it never counts the ratings of
    validation or test users."""
    with open(cfg.gt_path) as f:
        raw_gt: dict[str, list[str]] = json.load(f)
    counts: dict[str, int] = defaultdict(int)
    for uid in train_users:
        for a in set(raw_gt.get(uid, [])):
            counts[a] += 1
    return dict(counts)


def trivial_rankings(
    cfg: DomainConfig,
    idx_to_item: dict[int, str],
    train_users: list[str] | None = None,
    depth: int | None = None,
    legacy_popularity: bool = False,
) -> tuple[dict[str, list[str]], str | None]:
    """Constant ranked lists (top ``MAX_K``) of the non-personalised baselines.

    Returns ``({"popularity": [...], "top_category": [...]}, top_category_name)``.
    Items are ranked by ``train_popularity`` (ties by item id). The dominant
    category (taxonomy ``depth``, default ``cfg.category_depth``) is the one
    reached by most ``train_users`` through their consumed items; within it,
    items follow the popularity order. Nothing depends on the evaluated users.
    ``train_users=None`` falls back to the legacy ``cfg.canonical_train_path``;
    if that does not exist either, no list is returned.
    ``legacy_popularity=True`` ranks by the metadata ``rating_number`` instead,
    which includes the ratings of the evaluated users (only to reproduce the
    older protocol).
    """
    if train_users is None:
        if not cfg.canonical_train_path.exists():
            log.info(f"  train split not found ({cfg.canonical_train_path}); skipping baselines")
            return {}, None
        with open(cfg.canonical_train_path) as f:
            train_users = json.load(f)

    pop = item_popularity(cfg) if legacy_popularity else train_popularity(cfg, train_users)
    by_pop = sorted(set(idx_to_item.values()), key=lambda a: (-pop.get(a, 0), a))
    genres = item_genres(cfg, depth)
    counts = train_category_counts(cfg, train_users, depth)
    top_genre = max(counts, key=counts.get)
    rankings = {
        "popularity": by_pop[:MAX_K],
        "top_category": [a for a in by_pop if genres.get(a) == top_genre][:MAX_K],
    }
    return rankings, top_genre


def trivial_baselines(
    cfg: DomainConfig,
    gt: dict[str, set[str]],
    test_users: list[str],
    idx_to_item: dict[int, str],
    train_users: list[str] | None = None,
) -> list[dict]:
    """Popularity and constant top-genre baselines, both built from TRAINING
    users only (``train_users`` if given, else the legacy canonical train split
    of ``cfg``); empty when neither exists."""
    rankings, top_genre = trivial_rankings(cfg, idx_to_item, train_users)
    if top_genre is None:
        return []
    return [
        {"label": "Popularity", **evaluate_constant(rankings["popularity"], gt, test_users)},
        {
            "label": f"Top-genre ({top_genre})",
            **evaluate_constant(rankings["top_category"], gt, test_users),
        },
    ]


def run_all(
    cfg: DomainConfig,
    items_emb: np.ndarray,
    idx_to_item: dict[int, str],
    genre_gt: dict[str, set[str]],
    test_users: list[str],
    random_metrics: dict[str, float],
) -> list[dict]:
    """Evaluate every method registered for the active domain."""
    log.info("\nEvaluating all known methods...")
    log.info("  Legend: [LEAKAGE] = invalid result, do not report\n")
    all_results: list[dict] = [{"label": "Random", **random_metrics}]

    for label, rel_path, idx_rel, emb_rel, warn in KNOWN_METHODS.get(cfg.domain, []):
        path = PROCESSED / rel_path
        if not path.exists():
            log.info(f"  {label:<50}: not found")
            continue
        try:
            profiles = np.load(path).astype(np.float32)
            if idx_rel:
                with open(PROCESSED / idx_rel) as f:
                    idx_map: dict[str, int] = json.load(f)
            else:
                _, idx_map = load_profiles(path)

            # Array shorter than its index (stale artifact): skip with a clear
            # warning instead of failing with an IndexError.
            if idx_map and profiles.shape[0] <= max(idx_map.values()):
                log.warning(
                    f"  ⚠  {label}: misaligned array ({profiles.shape[0]:,} rows, "
                    f"index needs {max(idx_map.values()) + 1:,}) — stale, skipping"
                )
                continue

            method_emb = np.load(PROCESSED / emb_rel).astype(np.float32) if emb_rel else items_emb

            if warn:
                log.warning(f"  ⚠  {label}: {warn}")

            res = evaluate_method(
                label, profiles, method_emb, idx_map, idx_to_item, genre_gt, test_users
            )
            res.pop("_n_eval", None)
            all_results.append({"label": label, **res})

        except Exception as e:
            log.error(f"  {label}: ERROR — {e}")

    return all_results


def print_table(results: list[dict]) -> str:
    lines: list[str] = []
    lines.append("\n" + "=" * 75)
    lines.append("Results table")
    lines.append(
        f"Protocol: N={N_TEST_USERS}, seed={SEED}, MAX_BOOKS={MAX_BOOKS}, MAX_GENRES={MAX_GENRES}"
    )
    lines.append("  [LEAKAGE] invalid result, do not report")
    lines.append("=" * 75)
    header = f"  {'Method':<47} {'nDCG@10':>8} {'P@10':>8} {'R@10':>8} {'R@50':>8} {'R@100':>8}"
    lines.append(header)
    lines.append("  " + "─" * 71)

    for r in results:
        label = r["label"]
        vals = [r.get(m, 0) for m in ("nDCG@10", "P@10", "R@10", "R@50", "R@100")]
        lines.append(f"  {label:<47} " + " ".join(f"{v:>8.4f}" for v in vals))

    lines.append("=" * 75)
    return "\n".join(lines)


def save_canonical_csv(
    results: list[dict],
    timestamp: str,
    results_dir: Path | None = None,
) -> None:
    out_dir = results_dir or RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "canonical_results.csv"

    fieldnames = (
        ["label", "timestamp"]
        + [f"nDCG@{k}" for k in K_VALUES]
        + [f"P@{k}" for k in K_VALUES]
        + [f"R@{k}" for k in RECALL_K]
    )
    rows = [{"label": r["label"], "timestamp": timestamp, **r} for r in results]

    write_header = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
    log.info(f"  Canonical results saved: {path}")
