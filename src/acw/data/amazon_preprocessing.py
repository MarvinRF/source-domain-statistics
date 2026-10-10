"""
Standalone Books → CDs & Vinyl preprocessing of Amazon Reviews 2023 (the
original music-only script that produced the files in data/raw/; the generic
multi-domain version is ``acw.data.preprocessing``).

Usage (from the repository root):
    python -m acw.data.amazon_preprocessing

Outputs:
    data/raw/overlapping_users.json       — overlapping user_ids after 5-core
    data/raw/books_filtered.jsonl         — metadata of the retained books
    data/raw/albums_filtered.jsonl        — metadata of the retained albums
    data/raw/user_book_history.json       — per-user book history
    data/raw/ground_truth_real.json       — per-user relevant albums
    data/raw/preprocessing_stats.json     — run statistics

Approach:
    - Streaming reads (the raw files are never fully loaded into memory)
    - Iterative 5-core: keeps users/items with >= 5 interactions
    - Books act as the query side, albums as the ground truth
    - Rating >= 4 as a proxy for positive preference
"""

import gzip
import json
import os
from collections import defaultdict
from pathlib import Path

# Dataset locations — adjust to your local layout
DATA_SETS_DIR = Path(os.path.expanduser("~/Documentos/data-sets"))

BOOKS_REVIEWS_GZ = DATA_SETS_DIR / "livros" / "Books.jsonl.gz"
BOOKS_META_GZ = DATA_SETS_DIR / "livros" / "meta_Books.jsonl.gz"
MUSIC_REVIEWS_GZ = DATA_SETS_DIR / "musicas" / "CDs_and_Vinyl.jsonl.gz"
MUSIC_META_GZ = DATA_SETS_DIR / "musicas" / "meta_CDs_and_Vinyl.jsonl.gz"

RAW_DIR = Path("data/raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
MIN_RATING = 4  # minimum rating for a positive interaction
MIN_INTERACTIONS = 5  # 5-core: minimum interactions per user/item
MAX_USERS = None  # None = no limit; e.g. 50_000 for a quick debug run
LOG_INTERVAL = 500_000  # log progress every N lines read


# Utilities
def stream_jsonl_gz(path: Path, max_lines: int = None):
    """
    Stream a .jsonl.gz file line by line, yielding dicts. Malformed lines are
    silently skipped.
    """
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue
            count += 1
            if max_lines and count >= max_lines:
                break


def save_json(obj, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)
    print(f"  Saved: {path} ({os.path.getsize(path) / 1_048_576:.1f} MB)")


def save_jsonl(records: list[dict], path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  Saved: {path} ({os.path.getsize(path) / 1_048_576:.1f} MB)")


# Collect positive interactions (streaming)
def collect_interactions(reviews_gz: Path, min_rating: float = 4.0) -> tuple[dict, dict]:
    """
    Stream reviews and collect positive interactions (rating >= min_rating).

    Returns:
        user_to_items : dict {user_id: [(parent_asin, timestamp), ...]}
        item_to_users : dict {parent_asin: set(user_ids)}
    """
    user_to_items = defaultdict(list)
    item_to_users = defaultdict(set)
    total = 0
    kept = 0

    print(f"  Reading {reviews_gz.name}...")
    for record in stream_jsonl_gz(reviews_gz):
        total += 1
        if total % LOG_INTERVAL == 0:
            print(f"    {total:,} lines read, {kept:,} positive interactions kept")

        rating = record.get("rating", 0)
        user_id = record.get("user_id")
        asin = record.get("parent_asin") or record.get("asin")
        timestamp = record.get("timestamp", 0)

        if not user_id or not asin:
            continue
        if rating < min_rating:
            continue

        user_to_items[user_id].append((asin, timestamp))
        item_to_users[asin].add(user_id)
        kept += 1

    print(f"  Total: {total:,} reviews read, {kept:,} positive interactions")
    print(f"  Unique users: {len(user_to_items):,}")
    print(f"  Unique items: {len(item_to_users):,}")
    return dict(user_to_items), dict(item_to_users)


# Iterative k-core
def apply_kcore(
    user_to_items: dict,
    item_to_users: dict,
    k: int = 5,
) -> tuple[set, set]:
    """
    Iterative k-core filter: removes users with < k items and items with
    < k users until convergence.

    Returns:
        valid_users : set of valid user_ids
        valid_items : set of valid parent_asins
    """
    valid_users = set(user_to_items.keys())
    valid_items = set(item_to_users.keys())

    iteration = 0
    while True:
        iteration += 1
        prev_users = len(valid_users)
        prev_items = len(valid_items)

        # Drop users with too few valid items
        valid_users = {
            u
            for u in valid_users
            if sum(1 for asin, _ in user_to_items[u] if asin in valid_items) >= k
        }

        # Drop items with too few valid users
        valid_items = {i for i in valid_items if len(item_to_users[i] & valid_users) >= k}

        print(
            f"  [{iteration}] Users: {prev_users:,} → {len(valid_users):,} | "
            f"Items: {prev_items:,} → {len(valid_items):,}"
        )

        if len(valid_users) == prev_users and len(valid_items) == prev_items:
            break

    print(f"  5-core converged after {iteration} iterations")
    return valid_users, valid_items


# Overlapping users
def find_overlapping_users(
    book_valid_users: set,
    music_valid_users: set,
) -> set:
    """Return users present in both domains after 5-core filtering."""
    overlap = book_valid_users & music_valid_users
    print(f"  Overlapping users (books ∩ music): {len(overlap):,}")
    return overlap


# Per-user histories and ground truth
def build_histories(
    overlapping_users: set,
    book_user_to_items: dict,
    book_valid_items: set,
    music_user_to_items: dict,
    music_valid_items: set,
    max_users: int = None,
) -> tuple[dict, dict]:
    """
    For each overlapping user, build:
        book_history  : all positively rated books (sorted by timestamp)
        ground_truth  : positively rated albums (sorted by timestamp)

    No temporal cut-off is applied here. Users with < 3 books or < 3 albums
    are dropped.

    Returns:
        user_book_history : {user_id: [parent_asin, ...]}   (por timestamp asc)
        ground_truth      : {user_id: [parent_asin, ...]}
    """
    user_book_history = {}
    ground_truth = {}
    discarded = 0

    users_to_process = list(overlapping_users)
    if max_users:
        users_to_process = users_to_process[:max_users]

    for user_id in users_to_process:
        # The user's valid books, sorted by timestamp
        book_interactions = [
            (asin, ts)
            for asin, ts in book_user_to_items.get(user_id, [])
            if asin in book_valid_items
        ]
        book_interactions.sort(key=lambda x: x[1])
        books = [asin for asin, _ in book_interactions]

        # The user's valid albums, sorted by timestamp
        music_interactions = [
            (asin, ts)
            for asin, ts in music_user_to_items.get(user_id, [])
            if asin in music_valid_items
        ]
        music_interactions.sort(key=lambda x: x[1])
        albums = [asin for asin, _ in music_interactions]

        # Drop users with too short a history
        if len(books) < 3 or len(albums) < 3:
            discarded += 1
            continue

        user_book_history[user_id] = books
        ground_truth[user_id] = albums

    print(
        f"  Users with sufficient history (>= 3 books and >= 3 albums): {len(user_book_history):,}"
    )
    print(f"  Users dropped for short history: {discarded:,}")
    return user_book_history, ground_truth


# Metadata filtering
def prepare_item_text(record: dict, domain: str) -> str:
    """
    Build an item's descriptive text by concatenating the available fields,
    truncated to 1000 characters to bound the embedding input length.

    domain: "book" or "album"
    """
    parts = []

    title = record.get("title", "")
    if title:
        parts.append(f"Título: {title}")

    description = record.get("description", "")
    if isinstance(description, list):
        description = " ".join(description)
    if description:
        parts.append(str(description)[:600])

    categories = record.get("categories", [])
    if isinstance(categories, list) and categories:
        # Flatten nested category lists
        flat_cats = []
        for c in categories:
            if isinstance(c, list):
                flat_cats.extend(c)
            else:
                flat_cats.append(c)
        parts.append(f"Categorias: {', '.join(flat_cats[:8])}")

    features = record.get("features", [])
    if isinstance(features, list) and features:
        parts.append(" ".join(str(f) for f in features[:5]))

    text = " | ".join(p for p in parts if p.strip())
    return text[:1000] if text else ""


def filter_metadata(
    meta_gz: Path,
    relevant_asins: set,
    domain: str,
) -> list[dict]:
    """
    Stream metadata and return only the relevant items, with their text
    prepared for embedding.

    Returns a list of dicts with: parent_asin, title, text, average_rating,
    rating_number, categories
    """
    results = []
    total = 0
    found = 0

    print(f"  Reading {meta_gz.name} (filtering {len(relevant_asins):,} relevant ASINs)...")
    for record in stream_jsonl_gz(meta_gz):
        total += 1
        if total % LOG_INTERVAL == 0:
            print(f"    {total:,} metadata records read, {found:,} relevant found")

        asin = record.get("parent_asin")
        if not asin or asin not in relevant_asins:
            continue

        text = prepare_item_text(record, domain)
        if len(text) < 20:
            # Effectively empty metadata — skip
            continue

        results.append(
            {
                "parent_asin": asin,
                "title": record.get("title", ""),
                "text": text,
                "average_rating": record.get("average_rating"),
                "rating_number": record.get("rating_number"),
                "categories": record.get("categories", []),
            }
        )
        found += 1

        # Stop early once every relevant item has been found
        if found == len(relevant_asins):
            break

    print(f"  Total: {total:,} metadata records read, {found:,} relevant items with text")
    return results


# Final statistics
def compute_stats(
    user_book_history: dict,
    ground_truth: dict,
    books_meta: list,
    albums_meta: list,
) -> dict:
    book_lengths = [len(v) for v in user_book_history.values()]
    gt_lengths = [len(v) for v in ground_truth.values()]

    stats = {
        "n_users": len(user_book_history),
        "n_books": len(books_meta),
        "n_albums": len(albums_meta),
        "book_history": {
            "min": min(book_lengths),
            "max": max(book_lengths),
            "mean": round(sum(book_lengths) / len(book_lengths), 2),
        },
        "ground_truth": {
            "min": min(gt_lengths),
            "max": max(gt_lengths),
            "mean": round(sum(gt_lengths) / len(gt_lengths), 2),
            "total_pairs": sum(gt_lengths),
        },
    }
    return stats


# Main
if __name__ == "__main__":
    print("=" * 65)
    print("Amazon Reviews 2023 preprocessing (Books → CDs & Vinyl)")
    print("=" * 65)

    # ── Step 1: collect positive interactions ─────────────────────────────────
    print("\n[1/6] Collecting positive BOOKS interactions...")
    book_u2i, book_i2u = collect_interactions(BOOKS_REVIEWS_GZ, MIN_RATING)

    print("\n[2/6] Collecting positive MUSIC interactions...")
    music_u2i, music_i2u = collect_interactions(MUSIC_REVIEWS_GZ, MIN_RATING)

    # ── Step 2: 5-core ────────────────────────────────────────────────────────
    print("\n[3/6] Applying 5-core to BOOKS...")
    book_valid_users, book_valid_items = apply_kcore(book_u2i, book_i2u, MIN_INTERACTIONS)

    print("\n[3/6] Applying 5-core to MUSIC...")
    music_valid_users, music_valid_items = apply_kcore(music_u2i, music_i2u, MIN_INTERACTIONS)

    # ── Step 3: overlapping users ─────────────────────────────────────────────
    print("\n[4/6] Finding overlapping users...")
    overlapping_users = find_overlapping_users(book_valid_users, music_valid_users)

    if len(overlapping_users) < 100:
        print("\n  ⚠  WARNING: fewer than 100 overlapping users.")
        print("     Consider relaxing to 3-core (MIN_INTERACTIONS = 3).")

    # ── Step 4: histories and ground truth ────────────────────────────────────
    print("\n[5/6] Building histories and ground truth...")
    user_book_history, ground_truth = build_histories(
        overlapping_users=overlapping_users,
        book_user_to_items=book_u2i,
        book_valid_items=book_valid_items,
        music_user_to_items=music_u2i,
        music_valid_items=music_valid_items,
        max_users=MAX_USERS,
    )

    if len(user_book_history) == 0:
        print("\n  ✗ ERROR: no user has a sufficient history.")
        print("    Lower MIN_INTERACTIONS or MIN_RATING and try again.")
        exit(1)

    # ── Step 5: filter metadata of the relevant items ─────────────────────────
    print("\n[6/6] Filtering metadata of the relevant items...")

    # ASINs actually used in the histories
    relevant_book_asins = set(asin for books in user_book_history.values() for asin in books)
    relevant_album_asins = set(asin for albums in ground_truth.values() for asin in albums)

    print(f"  Relevant book ASINs:  {len(relevant_book_asins):,}")
    print(f"  Relevant album ASINs: {len(relevant_album_asins):,}")

    books_meta = filter_metadata(BOOKS_META_GZ, relevant_book_asins, "book")
    albums_meta = filter_metadata(MUSIC_META_GZ, relevant_album_asins, "album")

    # Drop albums without usable metadata from the ground truth
    albums_with_meta = {r["parent_asin"] for r in albums_meta}
    ground_truth_filtered = {
        uid: [a for a in albums if a in albums_with_meta] for uid, albums in ground_truth.items()
    }
    ground_truth_filtered = {
        uid: albums for uid, albums in ground_truth_filtered.items() if len(albums) >= 3
    }

    # Drop books without usable metadata from the histories
    books_with_meta = {r["parent_asin"] for r in books_meta}
    user_book_history_filtered = {
        uid: [b for b in books if b in books_with_meta]
        for uid, books in user_book_history.items()
        if uid in ground_truth_filtered
    }
    user_book_history_filtered = {
        uid: books for uid, books in user_book_history_filtered.items() if len(books) >= 3
    }

    # Keep only users present in both
    final_users = set(user_book_history_filtered) & set(ground_truth_filtered)
    user_book_history_final = {u: user_book_history_filtered[u] for u in final_users}
    ground_truth_final = {u: ground_truth_filtered[u] for u in final_users}

    print(f"\n  Final users after metadata filtering: {len(final_users):,}")

    # ── Save results ──────────────────────────────────────────────────────────
    print("\nSaving files...")
    save_json(list(final_users), RAW_DIR / "overlapping_users.json")
    save_json(user_book_history_final, RAW_DIR / "user_book_history.json")
    save_json(ground_truth_final, RAW_DIR / "ground_truth_real.json")
    save_jsonl(books_meta, RAW_DIR / "books_filtered.jsonl")
    save_jsonl(albums_meta, RAW_DIR / "albums_filtered.jsonl")

    stats = compute_stats(
        user_book_history_final,
        ground_truth_final,
        books_meta,
        albums_meta,
    )
    save_json(stats, RAW_DIR / "preprocessing_stats.json")

    # ── Final summary ─────────────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("SUMMARY — preprocessing finished")
    print("=" * 65)
    print(f"  Final users              : {stats['n_users']:,}")
    print(f"  Books with metadata      : {stats['n_books']:,}")
    print(f"  Albums with metadata     : {stats['n_albums']:,}")
    print(
        f"  Book history             : "
        f"min={stats['book_history']['min']}, "
        f"max={stats['book_history']['max']}, "
        f"mean={stats['book_history']['mean']}"
    )
    print(
        f"  Ground truth (albums)    : "
        f"min={stats['ground_truth']['min']}, "
        f"max={stats['ground_truth']['max']}, "
        f"mean={stats['ground_truth']['mean']}"
    )
    print(f"  Total user-album pairs   : {stats['ground_truth']['total_pairs']:,}")
    print("=" * 65)
    print("\nNext step: compute item embeddings (acw.embeddings.generate_embeddings)")
