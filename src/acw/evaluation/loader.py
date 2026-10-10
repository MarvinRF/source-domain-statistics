"""Data loading for evaluation: ground truth (exact or subgenre-expanded),
canonical evaluation users, item metadata and user profiles."""

from __future__ import annotations

import json
import random as _random
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from .config import MAX_GENRES, N_TEST_USERS, SEED, DomainConfig


@lru_cache(maxsize=8)
def _category_index(
    items_meta_path: Path, depth: int
) -> tuple[dict[str, str], dict[str, frozenset[str]]]:
    """Item -> category and category -> items at taxonomy level ``depth``
    (clamped to the deepest level an item has). Items without categories are
    ignored. Cached per (metadata file, depth); treat the result as read-only."""
    item_to_genre: dict[str, str] = {}
    genre_to_items: dict[str, set[str]] = defaultdict(set)
    with open(items_meta_path) as f:
        for line in f:
            item = json.loads(line)
            asin = item.get("parent_asin")
            cats: list[str] = item.get("categories", [])
            if not asin or len(cats) < 1:
                continue
            genre = cats[min(depth, len(cats) - 1)]
            item_to_genre[asin] = genre
            genre_to_items[genre].add(asin)
    return item_to_genre, {g: frozenset(v) for g, v in genre_to_items.items()}


@lru_cache(maxsize=2)
def _raw_exact(gt_path: Path) -> dict[str, list[str]]:
    with open(gt_path) as f:
        return json.load(f)


def build_genre_ground_truth(
    cfg: DomainConfig,
    users: set[str] | None = None,
    *,
    depth: int | None = None,
    max_genres: int | None = MAX_GENRES,
) -> dict[str, set[str]]:
    """Build the category-expanded ground truth.

    ``users`` restricts the output to those users (identical entries, far less
    memory on large catalogs such as movies, where the expanded sets of all
    users would not fit in RAM).

    Each user's exact target items are expanded to every catalog item of the
    same category (taxonomy level ``depth``, default ``cfg.category_depth``,
    clamped to the deepest available level). Items without categories are
    ignored. Users spanning more than ``max_genres`` distinct categories are
    excluded; ``max_genres=None`` keeps every user with at least one
    categorised item.
    """
    depth = cfg.category_depth if depth is None else depth
    gt_exact = _raw_exact(cfg.gt_path)
    item_to_genre, genre_to_items = _category_index(cfg.items_meta_path, depth)

    genre_gt: dict[str, set[str]] = {}
    for uid, items in gt_exact.items():
        if users is not None and uid not in users:
            continue
        genres = {item_to_genre[a] for a in items if a in item_to_genre}
        if not genres or (max_genres is not None and len(genres) > max_genres):
            continue
        if len(genres) == 1:  # share the category's item set instead of copying it
            genre_gt[uid] = genre_to_items[next(iter(genres))]
        else:
            genre_gt[uid] = frozenset().union(*(genre_to_items[g] for g in genres))

    return genre_gt


def build_exact_ground_truth(cfg: DomainConfig) -> dict[str, set[str]]:
    """Exact ground truth: the target items each user actually interacted with,
    restricted to the indexed catalog. No subgenre expansion, no MAX_GENRES
    filter."""
    with open(cfg.gt_path) as f:
        gt_exact: dict[str, list[str]] = json.load(f)
    with open(cfg.items_idx_path) as f:
        catalog = set(json.load(f))
    gt = {uid: set(items) & catalog for uid, items in gt_exact.items()}
    return {uid: items for uid, items in gt.items() if items}


def build_ground_truth(
    cfg: DomainConfig,
    mode: str = "genre",
    users: set[str] | None = None,
    *,
    depth: int | None = None,
    max_genres: int | None = MAX_GENRES,
) -> dict[str, set[str]]:
    """``mode="exact"`` (actual interactions) or ``"genre"`` (category
    expansion at ``depth``, optionally restricted to ``users``)."""
    if mode == "exact":
        return build_exact_ground_truth(cfg)
    if mode == "genre":
        return build_genre_ground_truth(cfg, users, depth=depth, max_genres=max_genres)
    raise ValueError(f"gt mode must be exact|genre, got {mode}")


def item_popularity(cfg: DomainConfig) -> dict[str, int]:
    """Per-item ``rating_number`` (Amazon metadata, independent of evaluated users)."""
    pop: dict[str, int] = {}
    with open(cfg.items_meta_path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("parent_asin"):
                pop[item["parent_asin"]] = item.get("rating_number") or 0
    return pop


def item_genres(cfg: DomainConfig, depth: int | None = None) -> dict[str, str]:
    """Category of each item at ``depth`` (default ``cfg.category_depth``),
    using the same rule as ``build_genre_ground_truth``."""
    depth = cfg.category_depth if depth is None else depth
    return dict(_category_index(cfg.items_meta_path, depth)[0])


def sample_test_users(
    genre_gt: dict[str, set[str]],
    cfg: DomainConfig,
) -> list[str]:
    """Return the canonical evaluation users that have ground truth.

    Priority:
      1. canonical_val_users.json  — validation split (~20% of eligible users)
      2. canonical_test_users.json — legacy fixed sample of 2,000 users
      3. Fallback: sample N_TEST_USERS with SEED (should not happen once
         generate_canonical_split has been run)
    """
    # Validation split (disjoint from training users)
    if cfg.canonical_val_path.exists():
        with open(cfg.canonical_val_path) as f:
            users: list[str] = json.load(f)
        return [u for u in users if u in genre_gt]

    # Legacy fixed sample
    if cfg.canonical_users_path.exists():
        with open(cfg.canonical_users_path) as f:
            users = json.load(f)
        return [u for u in users if u in genre_gt]

    # Fallback
    history_path = cfg.raw_dir / "user_book_history.json"
    with open(history_path) as f:
        book_history: dict[str, list] = json.load(f)

    eligible = list(set(book_history.keys()) & set(genre_gt.keys()))
    _random.seed(SEED)
    return _random.sample(eligible, min(N_TEST_USERS, len(eligible)))


def load_profiles(path: Path) -> tuple[np.ndarray, dict[str, int]]:
    """Load a .npy profile matrix and infer its matching index file."""
    profiles = np.load(path).astype(np.float32)

    idx_path = path.with_name(path.stem.replace("user_profiles_", "user_profile_index_") + ".json")
    if not idx_path.exists():
        for suffix in [
            "_pw",
            "_whitened",
            "_profile_whitened",
            "_infomax",
            "_moe",
            "_nlpw",
            "_mpw",
            "_pw_zb",
            "_nlpw_zb",
            "_moe_zb",
            "_two_stage",
        ]:
            candidate = path.with_name(
                path.stem.replace(suffix, "").replace("user_profiles_", "user_profile_index_")
                + ".json"
            )
            if candidate.exists():
                idx_path = candidate
                break

    with open(idx_path) as f:
        idx_map: dict[str, int] = json.load(f)
    return profiles, idx_map
