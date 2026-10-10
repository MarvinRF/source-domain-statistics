# src/acw/data/preprocessing/pipeline.py
"""
Core Amazon Reviews 2023 preprocessing for any Books → X domain pair:
positive-interaction collection, iterative k-core filtering, per-user
histories and ground truth, and metadata filtering. Run it through the CLI:

    python -m acw.data.preprocessing.cli preprocess --target-domain music
"""

from __future__ import annotations

import logging
from pathlib import Path

from .config import LOG_INTERVAL
from .io import stream_jsonl_gz
from .models import (
    DomainPaths,
    HistogramStats,
    InteractionData,
    ItemRecord,
    ItemToUsers,
    KCoreResult,
    PreprocessingConfig,
    PreprocessingResult,
    UserToItems,
)

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Collect positive interactions
# ─────────────────────────────────────────────────────────────────────────────


def collect_interactions(
    reviews_gz: Path,
    min_rating: float,
    log_interval: int = LOG_INTERVAL,
) -> InteractionData:
    """
    Stream reviews and collect positive interactions (rating >= min_rating),
    keeping timestamps for later temporal ordering.
    """
    user_to_items: UserToItems = {}
    item_to_users: ItemToUsers = {}
    total = 0
    kept = 0

    log.info(f"  Reading {reviews_gz.name}...")
    for record in stream_jsonl_gz(reviews_gz):
        total += 1
        if total % log_interval == 0:
            log.info(f"    {total:,} lines read | {kept:,} positive interactions")

        rating = float(record.get("rating", 0))
        user_id = record.get("user_id")
        asin = record.get("parent_asin") or record.get("asin")
        timestamp = int(record.get("timestamp", 0))

        if not user_id or not asin or rating < min_rating:
            continue

        if user_id not in user_to_items:
            user_to_items[user_id] = []
        user_to_items[user_id].append((asin, timestamp))

        if asin not in item_to_users:
            item_to_users[asin] = set()
        item_to_users[asin].add(user_id)

        kept += 1

    log.info(
        f"  {total:,} reviews | {kept:,} positive | "
        f"{len(user_to_items):,} users | {len(item_to_users):,} items"
    )

    return InteractionData(
        user_to_items=user_to_items,
        item_to_users=item_to_users,
        total_reviews=total,
        kept_reviews=kept,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 2. Iterative k-core filter
# ─────────────────────────────────────────────────────────────────────────────


def apply_kcore(
    data: InteractionData,
    k: int,
) -> KCoreResult:
    """
    Apply an iterative k-core filter until convergence, removing users with
    fewer than k valid items and items with fewer than k valid users.
    """
    valid_users: set[str] = set(data.user_to_items)
    valid_items: set[str] = set(data.item_to_users)
    iteration = 0

    while True:
        iteration += 1
        prev_users = len(valid_users)
        prev_items = len(valid_items)

        valid_users = {
            u
            for u in valid_users
            if sum(1 for asin, _ in data.user_to_items[u] if asin in valid_items) >= k
        }
        valid_items = {i for i in valid_items if len(data.item_to_users[i] & valid_users) >= k}

        log.info(
            f"  [{iteration}] Users: {prev_users:,} → {len(valid_users):,} | "
            f"Items: {prev_items:,} → {len(valid_items):,}"
        )

        if len(valid_users) == prev_users and len(valid_items) == prev_items:
            break

    log.info(f"  k-core converged after {iteration} iteration(s)")
    return KCoreResult(
        valid_users=valid_users,
        valid_items=valid_items,
        iterations=iteration,
    )


# ─────────────────────────────────────────────────────────────────────────────
# 3. Per-user histories and ground truth
# ─────────────────────────────────────────────────────────────────────────────


def build_histories(
    overlapping_users: set[str],
    source_data: InteractionData,
    source_valid_items: set[str],
    target_data: InteractionData,
    target_valid_items: set[str],
    min_history: int,
    max_users: int | None = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    For each overlapping user, build:
        book_history : positively rated books (timestamp ascending)
        ground_truth : target-domain items (timestamp ascending)

    Users with fewer than min_history items in either domain are dropped.
    No temporal cut-off is applied here.
    """
    book_history: dict[str, list[str]] = {}
    ground_truth: dict[str, list[str]] = {}
    discarded = 0

    users = list(overlapping_users)
    if max_users is not None:
        users = users[:max_users]

    for uid in users:
        # Valid books, sorted by timestamp
        src_interactions = sorted(
            (
                (asin, ts)
                for asin, ts in source_data.user_to_items.get(uid, [])
                if asin in source_valid_items
            ),
            key=lambda x: x[1],
        )
        books = [asin for asin, _ in src_interactions]

        # Valid target items, sorted by timestamp
        tgt_interactions = sorted(
            (
                (asin, ts)
                for asin, ts in target_data.user_to_items.get(uid, [])
                if asin in target_valid_items
            ),
            key=lambda x: x[1],
        )
        targets = [asin for asin, _ in tgt_interactions]

        if len(books) < min_history or len(targets) < min_history:
            discarded += 1
            continue

        book_history[uid] = books
        ground_truth[uid] = targets

    log.info(
        f"  Users with history >= {min_history}: {len(book_history):,} | Dropped: {discarded:,}"
    )
    return book_history, ground_truth


# ─────────────────────────────────────────────────────────────────────────────
# 4. Item text for embedding
# ─────────────────────────────────────────────────────────────────────────────


def _build_item_text(record: dict) -> str:
    """
    Concatenate the available metadata fields into a descriptive text,
    truncated to 1,000 characters to bound the embedding input length.
    """
    parts: list[str] = []

    title = str(record.get("title", "")).strip()
    if title:
        parts.append(title)

    description = record.get("description", "")
    if isinstance(description, list):
        description = " ".join(str(d) for d in description if d)
    description = str(description).strip()
    if description:
        parts.append(description[:600])

    categories = record.get("categories", [])
    if isinstance(categories, list) and categories:
        flat: list[str] = []
        for c in categories:
            if isinstance(c, list):
                flat.extend(str(x) for x in c)
            else:
                flat.append(str(c))
        if flat:
            parts.append(", ".join(flat[:8]))

    features = record.get("features", [])
    if isinstance(features, list) and features:
        parts.append(" ".join(str(f) for f in features[:5]))

    text = " | ".join(p for p in parts if p.strip())
    return text[:1_000]


# ─────────────────────────────────────────────────────────────────────────────
# 5. Metadata filtering
# ─────────────────────────────────────────────────────────────────────────────


def filter_metadata(
    meta_gz: Path,
    relevant_asins: set[str],
    min_text_len: int,
    log_interval: int = LOG_INTERVAL,
) -> list[ItemRecord]:
    """
    Stream metadata and return only items in relevant_asins whose prepared
    text has at least min_text_len characters.
    """
    results: list[ItemRecord] = []
    total = 0
    found = 0

    log.info(f"  Reading {meta_gz.name} ({len(relevant_asins):,} expected ASINs)...")
    for record in stream_jsonl_gz(meta_gz):
        total += 1
        if total % log_interval == 0:
            log.info(f"    {total:,} metadata records | {found:,} relevant")

        asin = record.get("parent_asin")
        if not asin or asin not in relevant_asins:
            continue

        text = _build_item_text(record)
        if len(text) < min_text_len:
            continue

        rating_raw = record.get("average_rating")
        rating_n = record.get("rating_number")

        results.append(
            ItemRecord(
                parent_asin=asin,
                title=str(record.get("title", "")).strip(),
                text=text,
                average_rating=float(rating_raw) if rating_raw is not None else None,
                rating_number=int(rating_n) if rating_n is not None else None,
                categories=[str(c) for c in record.get("categories", [])],
            )
        )
        found += 1

        if found == len(relevant_asins):
            break  # all found — stop early

    log.info(f"  {total:,} metadata records read | {found:,} items with valid text")
    return results


# ─────────────────────────────────────────────────────────────────────────────
# 6. Final synchronisation and statistics
# ─────────────────────────────────────────────────────────────────────────────


def _sync_and_filter(
    book_history: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    books_with_meta: set[str],
    target_with_meta: set[str],
    min_history: int,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Drop items without metadata from the histories, drop users that fall
    below min_history afterwards, and keep the same user_ids in both dicts.
    """
    gt_clean = {
        uid: [a for a in albums if a in target_with_meta] for uid, albums in ground_truth.items()
    }
    gt_clean = {uid: v for uid, v in gt_clean.items() if len(v) >= min_history}

    bh_clean = {
        uid: [b for b in books if b in books_with_meta]
        for uid, books in book_history.items()
        if uid in gt_clean
    }
    bh_clean = {uid: v for uid, v in bh_clean.items() if len(v) >= min_history}

    final_users = set(bh_clean) & set(gt_clean)
    return (
        {u: bh_clean[u] for u in final_users},
        {u: gt_clean[u] for u in final_users},
    )


def _histogram(values: list[int]) -> HistogramStats:
    return HistogramStats(
        min=min(values),
        max=max(values),
        mean=round(sum(values) / len(values), 2),
        total=sum(values),
    )


def _compute_stats(
    book_history: dict[str, list[str]],
    ground_truth: dict[str, list[str]],
    books_meta: list[ItemRecord],
    target_meta: list[ItemRecord],
) -> dict[str, object]:
    bh = _histogram([len(v) for v in book_history.values()])
    gt = _histogram([len(v) for v in ground_truth.values()])
    return {
        "n_users": len(book_history),
        "n_books": len(books_meta),
        "n_target": len(target_meta),
        "book_history": {"min": bh.min, "max": bh.max, "mean": bh.mean},
        "ground_truth": {
            "min": gt.min,
            "max": gt.max,
            "mean": gt.mean,
            "total_pairs": gt.total,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────


def run(paths: DomainPaths, cfg: PreprocessingConfig) -> PreprocessingResult:
    """
    Run the full preprocessing pipeline for a domain pair.

    Args:
        paths : resolved paths for the source (Books) and target (Music/Movies/Games)
        cfg   : run parameters

    Returns:
        PreprocessingResult with histories, ground truth, metadata and stats
    """
    log.info("=" * 65)
    log.info(f"Preprocessing: Books → {paths.target_domain.upper()}")
    log.info("=" * 65)

    # ── 1. Positive interactions ──────────────────────────────────────────────
    log.info("\n[1/5] Collecting BOOKS interactions...")
    source_data = collect_interactions(paths.source_reviews, cfg.min_rating, cfg.log_interval)

    log.info(f"\n[2/5] Collecting {paths.target_domain.upper()} interactions...")
    target_data = collect_interactions(paths.target_reviews, cfg.min_rating, cfg.log_interval)

    # ── 2. k-core ─────────────────────────────────────────────────────────────
    log.info(f"\n[3/5] Applying {cfg.min_interactions}-core to BOOKS...")
    source_kcore = apply_kcore(source_data, cfg.min_interactions)

    log.info(f"\n[3/5] Applying {cfg.min_interactions}-core to {paths.target_domain.upper()}...")
    target_kcore = apply_kcore(target_data, cfg.min_interactions)

    # ── 3. Overlapping users ──────────────────────────────────────────────────
    overlapping = source_kcore.valid_users & target_kcore.valid_users
    log.info(f"\n[4/5] Overlapping users (Books ∩ {paths.target_domain}): {len(overlapping):,}")
    if len(overlapping) < 100:
        log.warning("  ⚠ Fewer than 100 overlapping users — consider lowering k.")

    # ── 4. Histories and ground truth ─────────────────────────────────────────
    log.info("\n[4/5] Building histories and ground truth...")
    book_history, ground_truth = build_histories(
        overlapping_users=overlapping,
        source_data=source_data,
        source_valid_items=source_kcore.valid_items,
        target_data=target_data,
        target_valid_items=target_kcore.valid_items,
        min_history=cfg.min_history,
        max_users=cfg.max_users,
    )

    if not book_history:
        raise RuntimeError(
            "No user has a sufficient history. Lower min_interactions or min_rating."
        )

    # ── 5. Metadata ──────────────────────────────────────────────────────────
    log.info("\n[5/5] Filtering metadata...")

    relevant_books = {a for books in book_history.values() for a in books}
    relevant_target = {a for items in ground_truth.values() for a in items}
    log.info(f"  Relevant book ASINs:   {len(relevant_books):,}")
    log.info(f"  Relevant target ASINs: {len(relevant_target):,}")

    books_meta = filter_metadata(
        paths.source_meta, relevant_books, cfg.min_text_len, cfg.log_interval
    )
    target_meta = filter_metadata(
        paths.target_meta, relevant_target, cfg.min_text_len, cfg.log_interval
    )

    # ── 6. Final synchronisation ──────────────────────────────────────────────
    books_with_meta = {r.parent_asin for r in books_meta}
    target_with_meta = {r.parent_asin for r in target_meta}

    book_history_final, ground_truth_final = _sync_and_filter(
        book_history,
        ground_truth,
        books_with_meta,
        target_with_meta,
        cfg.min_history,
    )
    log.info(f"  Final users: {len(book_history_final):,}")

    stats = _compute_stats(book_history_final, ground_truth_final, books_meta, target_meta)

    return PreprocessingResult(
        user_book_history=book_history_final,
        ground_truth=ground_truth_final,
        books_meta=books_meta,
        target_meta=target_meta,
        stats=stats,
    )
