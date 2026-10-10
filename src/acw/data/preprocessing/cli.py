# src/acw/data/preprocessing/cli.py
"""
Command-line entry point for Amazon Reviews 2023 preprocessing.

Subcommands:
    preprocess      Run the full preprocessing pipeline.
    build-canonical Sample canonical_test_users_<domain>.json from the preprocess outputs.

Works for any Books → X domain pair. The target domain is selected with
--target-domain and the dataset location with --datasets-dir.

Outputs (in data/raw/ for music, data/raw/<domain>/ otherwise):
    overlapping_users.json    — eligible user_ids after k-core and filtering
    user_book_history.json    — {user_id: [book_asin, ...]}
    ground_truth_real.json    — {user_id: [target_asin, ...]}
    books_filtered.jsonl      — metadata of the retained books
    <target>_filtered.jsonl   — metadata of the retained target items
    preprocessing_stats.json  — run statistics

Usage:
    # Books → Music, default paths
    python -m acw.data.preprocessing.cli preprocess --target-domain music

    # Books → Movies
    python -m acw.data.preprocessing.cli preprocess --target-domain movies

    # Explicit paths
    python -m acw.data.preprocessing.cli preprocess \\
        --target-domain games \\
        --datasets-dir $ACW_DATASETS_DIR \\
        --out-dir data/raw/games

    # Quick debug run
    python -m acw.data.preprocessing.cli preprocess \\
        --target-domain movies --max-users 5000 --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
from pathlib import Path

from .config import (
    DATA_SETS_DIR,
    KNOWN_DOMAINS,
    MIN_HISTORY,
    MIN_INTERACTIONS,
    MIN_RATING,
    RAW_DIR,
    ROOT,
)
from .io import save_json, save_jsonl
from .models import DomainPaths, PreprocessingConfig
from .pipeline import run

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: preprocess
# ─────────────────────────────────────────────────────────────────────────────


def _add_preprocess_args(sub: argparse.ArgumentParser) -> None:
    """Register the arguments of the preprocess subcommand."""
    sub.add_argument(
        "--target-domain",
        required=True,
        choices=list(KNOWN_DOMAINS),
        help="Target domain of the cross-domain recommendation.",
    )
    sub.add_argument(
        "--datasets-dir",
        type=Path,
        default=DATA_SETS_DIR,
        help=f"Root directory of the .gz datasets (default: {DATA_SETS_DIR})",
    )
    sub.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Output directory for the processed files. "
            "Default: data/raw/ for music, data/raw/<domain>/ otherwise."
        ),
    )
    sub.add_argument("--min-rating", type=float, default=MIN_RATING)
    sub.add_argument("--min-interactions", type=int, default=MIN_INTERACTIONS)
    sub.add_argument("--min-history", type=int, default=MIN_HISTORY)
    sub.add_argument("--max-users", type=int, default=None)
    sub.add_argument("--dry-run", action="store_true")


def _run_preprocess(args: argparse.Namespace) -> None:
    paths = _resolve_paths(args)
    cfg = PreprocessingConfig(
        min_rating=args.min_rating,
        min_interactions=args.min_interactions,
        min_history=args.min_history,
        max_users=args.max_users,
    )
    result = run(paths, cfg)

    if args.dry_run:
        log.info("\n[DRY RUN] No files written.")
    else:
        log.info("\nSaving files...")
        out = paths.out_dir
        domain = paths.target_domain
        save_json(list(result.user_book_history.keys()), out / "overlapping_users.json")
        save_json(result.user_book_history, out / "user_book_history.json")
        save_json(result.ground_truth, out / "ground_truth_real.json")
        save_jsonl([vars(r) for r in result.books_meta], out / "books_filtered.jsonl")
        save_jsonl([vars(r) for r in result.target_meta], out / f"{domain}_filtered.jsonl")
        save_json(result.stats, out / "preprocessing_stats.json")

    s = result.stats
    log.info("\n" + "=" * 65)
    log.info(f"SUMMARY — Books → {paths.target_domain.upper()}")
    log.info("=" * 65)
    log.info(f"  Final users              : {s['n_users']:,}")
    log.info(f"  Books with metadata      : {s['n_books']:,}")
    log.info(f"  Target items w/ metadata : {s['n_target']:,}")
    bh = s["book_history"]
    gt = s["ground_truth"]
    log.info(f"  Book history             : min={bh['min']} max={bh['max']} mean={bh['mean']}")
    log.info(f"  Ground truth             : min={gt['min']} max={gt['max']} mean={gt['mean']}")
    log.info(f"  Total user-item pairs    : {gt['total_pairs']:,}")
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Subcommand: build-canonical
# ─────────────────────────────────────────────────────────────────────────────

# Canonical number of test users per domain (games has a smaller user base).
_CANONICAL_N: dict[str, int] = {"music": 2000, "movies": 2000, "games": 1000}

# Fixed seed; changing it changes the sampled users.
_CANONICAL_SEED = 42


def _add_build_canonical_args(sub: argparse.ArgumentParser) -> None:
    sub.add_argument(
        "--target-domain",
        required=True,
        choices=list(KNOWN_DOMAINS),
        help="Domain for which to sample the canonical test users.",
    )
    sub.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing overlapping_users.json and ground_truth_real.json. "
            "Default: data/raw/ for music, data/raw/<domain>/ otherwise."
        ),
    )
    sub.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: data/processed/",
    )
    sub.add_argument(
        "--n",
        type=int,
        default=None,
        help=f"Number of canonical users. Per-domain defaults: {_CANONICAL_N}",
    )
    sub.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing file (default: abort if it exists).",
    )


def _run_build_canonical(args: argparse.Namespace) -> None:
    """
    Sample canonical_test_users_<domain>.json.

    A user is eligible if present in both overlapping_users.json and
    ground_truth_real.json (i.e. has target items after k-core filtering).
    MAX_GENRES filtering is left to the evaluation module.

    The music file (canonical_test_users.json) is treated as a fixed
    reference and is never overwritten without --force.
    """
    domain = args.target_domain
    processed_dir = ROOT / "data" / "processed"

    # Protect the existing music reference file
    if domain == "music":
        music_canonical = processed_dir / "canonical_test_users.json"
        if music_canonical.exists() and not args.force:
            log.error(
                "canonical_test_users.json (music) already exists and is the "
                "fixed reference sample. Use --force only if you intend to "
                "replace it — previously reported numbers would no longer be comparable."
            )
            sys.exit(1)

    # Resolve the directory holding the preprocessing outputs
    if args.raw_dir is not None:
        raw_dir = args.raw_dir
    elif domain == "music":
        raw_dir = RAW_DIR
    else:
        raw_dir = RAW_DIR / domain

    overlapping_path = raw_dir / "overlapping_users.json"
    gt_path = raw_dir / "ground_truth_real.json"

    for p in (overlapping_path, gt_path):
        if not p.exists():
            log.error(
                f"File not found: {p}\n"
                f"Run first: python -m acw.data.preprocessing.cli preprocess "
                f"--target-domain {domain}"
            )
            sys.exit(1)

    with open(overlapping_path) as f:
        overlapping: list[str] = json.load(f)
    with open(gt_path) as f:
        gt: dict = json.load(f)

    gt_set = set(gt.keys())
    eligible = [u for u in overlapping if u in gt_set]

    n = args.n if args.n is not None else _CANONICAL_N.get(domain, 2000)
    if len(eligible) < n:
        log.warning(
            f"Only {len(eligible):,} eligible users — using all of them (requested: {n:,})."
        )
        n = len(eligible)

    rng = random.Random(_CANONICAL_SEED)
    canonical = rng.sample(eligible, n)

    # Output file
    out_dir = args.out_dir if args.out_dir is not None else processed_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if domain == "music":
        out_path = out_dir / "canonical_test_users.json"
    else:
        out_path = out_dir / f"canonical_test_users_{domain}.json"

    if out_path.exists() and not args.force:
        log.error(f"{out_path.name} already exists. Use --force to overwrite.")
        sys.exit(1)

    save_json(canonical, out_path)

    # Report overlap with the music sample (informational only)
    music_canonical_path = processed_dir / "canonical_test_users.json"
    if music_canonical_path.exists() and domain != "music":
        with open(music_canonical_path) as f:
            music_set = set(json.load(f))
        overlap = len(set(canonical) & music_set)
        log.info(
            f"  Overlap with the music canonical sample: {overlap:,} users "
            f"(expected — different domains, trained separately)"
        )

    log.info("=" * 65)
    log.info(f"CANONICAL TEST USERS — {domain.upper()}")
    log.info("=" * 65)
    log.info(f"  Eligible   : {len(eligible):,}")
    log.info(f"  Sampled    : {n:,}  (seed={_CANONICAL_SEED})")
    log.info(f"  File       : {out_path}")
    log.info("=" * 65)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point with subparsers
# ─────────────────────────────────────────────────────────────────────────────


def _resolve_paths(args: argparse.Namespace) -> DomainPaths:
    """Build DomainPaths from the CLI arguments."""
    datasets_dir: Path = args.datasets_dir
    domain: str = args.target_domain

    subdir, reviews_file, meta_file, label = KNOWN_DOMAINS[domain]

    # Books (source domain)
    source_reviews = datasets_dir / "livros" / "Books.jsonl.gz"
    source_meta = datasets_dir / "livros" / "meta_Books.jsonl.gz"

    # Target domain
    target_reviews = datasets_dir / subdir / reviews_file
    target_meta = datasets_dir / subdir / meta_file

    # Fail early if any input file is missing
    for path in (source_reviews, source_meta, target_reviews, target_meta):
        if not path.exists():
            log.error(f"File not found: {path}")
            sys.exit(1)

    # Output directory
    if args.out_dir is not None:
        out_dir = args.out_dir
    elif domain == "music":
        out_dir = RAW_DIR  # music outputs live directly in data/raw/
    else:
        out_dir = RAW_DIR / domain

    out_dir.mkdir(parents=True, exist_ok=True)

    return DomainPaths(
        source_reviews=source_reviews,
        source_meta=source_meta,
        target_reviews=target_reviews,
        target_meta=target_meta,
        out_dir=out_dir,
        target_label=label,
        target_domain=domain,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Amazon Reviews 2023 preprocessing — Books → X",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subs = p.add_subparsers(dest="subcommand", required=True)

    _add_preprocess_args(
        subs.add_parser(
            "preprocess",
            help="Run the full preprocessing pipeline.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )
    _add_build_canonical_args(
        subs.add_parser(
            "build-canonical",
            help="Sample canonical_test_users_<domain>.json.",
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
    )

    args = p.parse_args()

    if args.subcommand == "preprocess":
        _run_preprocess(args)
    elif args.subcommand == "build-canonical":
        _run_build_canonical(args)


if __name__ == "__main__":
    main()
