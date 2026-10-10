"""
Generate the canonical 80/20 train/val user split (music) over all eligible
users: users with an H3 profile ∩ users with music ground truth spanning at
most 6 subgenres.

The split is deterministic (np.random.default_rng(42).permutation); val is the
last 20% of the permutation, matching the alignment training scripts.

Output:
  data/processed/canonical_train_users.json  (80%, ~11k users)
  data/processed/canonical_val_users.json    (20%, ~2.7k users)

Usage:
    python -m acw.data.preprocessing.generate_canonical_split
    python -m acw.data.preprocessing.generate_canonical_split --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import sys

import numpy as np

from acw.settings import ROOT  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SEED = 42
MAX_GENRES = 6
CAT_DEPTH = 2


def build_music_gt() -> dict[str, set[str]]:
    from collections import defaultdict

    with open(RAW / "ground_truth_real.json") as f:
        gt_exact: dict[str, list[str]] = json.load(f)

    item_to_genre: dict[str, str] = {}
    genre_to_items: dict[str, set[str]] = defaultdict(set)

    with open(RAW / "albums_filtered.jsonl") as f:
        for line in f:
            item = json.loads(line)
            asin = item.get("parent_asin")
            cats: list[str] = item.get("categories", [])
            if not asin or len(cats) < 1:
                continue
            genre = cats[min(CAT_DEPTH, len(cats) - 1)]
            item_to_genre[asin] = genre
            genre_to_items[genre].add(asin)

    genre_gt: dict[str, set[str]] = {}
    for uid, items in gt_exact.items():
        genres = {item_to_genre[a] for a in items if a in item_to_genre}
        if not genres or len(genres) > MAX_GENRES:
            continue
        relevant: set[str] = set()
        for g in genres:
            relevant |= genre_to_items[g]
        genre_gt[uid] = relevant

    return genre_gt


def main(dry_run: bool = False) -> None:
    log.info("=" * 70)
    log.info("Canonical split generation (music)")
    log.info(f"  SEED={SEED}  |  val=last 20%  |  dry_run={dry_run}")
    log.info("=" * 70)

    # ── Users with a valid H3 profile ─────────────────────────────────────────
    h3_idx_path = PROC / "user_profile_index_h3.json"
    if not h3_idx_path.exists():
        log.error(f"Not found: {h3_idx_path}")
        log.error("Generate and embed the H3 user profiles first.")
        sys.exit(1)

    with open(h3_idx_path) as f:
        h3_idx: dict[str, int] = json.load(f)
    log.info(f"  Users with an H3 profile: {len(h3_idx):,}")

    # ── Music ground truth ────────────────────────────────────────────────────
    log.info("\nBuilding music ground truth...")
    genre_gt = build_music_gt()
    log.info(f"  Users with music GT (≤{MAX_GENRES} subgenres): {len(genre_gt):,}")

    # ── Eligible: H3 ∩ GT ─────────────────────────────────────────────────────
    eligible = sorted(
        (u for u in h3_idx if u in genre_gt),
        key=lambda u: h3_idx[u],
    )
    log.info(f"  Eligible (H3 ∩ GT): {len(eligible):,}")

    # ── Split 80/20 ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(eligible))
    n_train = int(len(eligible) * 0.8)

    train_users = [eligible[i] for i in perm[:n_train]]
    val_users = [eligible[i] for i in perm[n_train:]]

    log.info(f"\nSplit 80/20 (seed={SEED}):")
    log.info(f"  Train: {len(train_users):,}")
    log.info(f"  Val:   {len(val_users):,}")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    train_set = set(train_users)
    val_set = set(val_users)
    assert not train_set & val_set, "ERROR: train/val overlap!"
    assert len(train_set) + len(val_set) == len(eligible), "ERROR: users lost!"
    log.info("  Sanity: no train/val overlap ✓")

    if dry_run:
        log.info("\n[DRY RUN] — no files written.")
        return

    # ── Save ──────────────────────────────────────────────────────────────────
    train_path = PROC / "canonical_train_users.json"
    val_path = PROC / "canonical_val_users.json"

    with open(train_path, "w") as f:
        json.dump(train_users, f)
    with open(val_path, "w") as f:
        json.dump(val_users, f)

    log.info(f"\nSaved: {train_path}")
    log.info(f"Saved: {val_path}")
    log.info("\nNext step:")
    log.info("  python -m acw.evaluation.evaluate --domain music --all")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()
    main(dry_run=args.dry_run)
