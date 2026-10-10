"""Command-line interface for the canonical zero-shot cross-domain evaluation.

Usage:
    # Music (default) — all registered methods
    python -m acw.evaluation.evaluate --all

    # Music — a single method
    python -m acw.evaluation.evaluate \\
        --profiles data/processed/alignment/user_profiles_h3_pw_zb.npy \\
        --label "H3+PW-ZB"

    # Movies — all registered methods
    python -m acw.evaluation.evaluate --domain movies --all

    # Any domain with explicit path overrides
    python -m acw.evaluation.evaluate \\
        --domain movies \\
        --gt-path data/raw/movies/ground_truth_real.json \\
        --items-emb-path data/processed/movie_embeddings.npy \\
        --profiles data/processed/user_profiles_h3_movies.npy \\
        --label "H3-Movies"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from .config import (
    K_VALUES,
    KNOWN_DOMAINS,
    MAX_BOOKS,
    MAX_GENRES,
    N_TEST_USERS,
    PROCESSED,
    RESULTS_DIR,
    SEED,
    resolve_paths,
)
from .loader import build_ground_truth, load_profiles, sample_test_users
from .pipeline import (
    compute_random_baseline,
    evaluate_method,
    print_table,
    run_all,
    save_canonical_csv,
    trivial_baselines,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Canonical evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--domain",
        choices=sorted(KNOWN_DOMAINS),
        default="music",
        help="Target domain (default: music). Sets the default paths for every field.",
    )
    # ── Single method ────────────────────────────────────────────────────────
    p.add_argument(
        "--profiles", type=str, default=None, help="Path to the .npy user profiles to evaluate."
    )
    p.add_argument(
        "--index",
        type=str,
        default=None,
        help="Path to the profile index .json (inferred if omitted).",
    )
    p.add_argument(
        "--album-emb",
        type=str,
        default=None,
        help="Path to the item embeddings (alias of --items-emb-path).",
    )
    p.add_argument("--label", type=str, default=None, help="Method label in the results table.")
    # ── Batch mode ───────────────────────────────────────────────────────────
    p.add_argument(
        "--all", action="store_true", help="Evaluate every method registered for the active domain."
    )
    p.add_argument(
        "--gt",
        choices=("exact", "genre"),
        default="genre",
        help="exact = items actually consumed; genre = subgenre expansion.",
    )
    p.add_argument(
        "--no-save", action="store_true", help="Do not write the CSV or the summary file."
    )
    # ── Path overrides ───────────────────────────────────────────────────────
    # --domain sets the defaults; each argument below overrides one field.
    p.add_argument(
        "--gt-path",
        type=str,
        default=None,
        help="Override: path to the domain's ground-truth JSON.",
    )
    p.add_argument(
        "--items-emb-path",
        type=str,
        default=None,
        help="Override: path to the target item embeddings (.npy).",
    )
    p.add_argument(
        "--items-idx-path",
        type=str,
        default=None,
        help="Override: path to the item id_to_idx .json.",
    )
    p.add_argument(
        "--canonical-users-path",
        type=str,
        default=None,
        help="Override: path to the domain's canonical_test_users.json.",
    )
    p.add_argument(
        "--items-meta-path",
        type=str,
        default=None,
        help="Override: path to the item metadata .jsonl.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    cfg = resolve_paths(
        domain=args.domain,
        gt_path=args.gt_path,
        # --items-emb-path takes precedence over --album-emb
        items_emb_path=args.items_emb_path or args.album_emb,
        items_idx_path=args.items_idx_path,
        canonical_users_path=args.canonical_users_path,
        items_meta_path=args.items_meta_path,
    )

    log.info("=" * 75)
    log.info(f"Canonical evaluation  [domain={cfg.domain}]")
    log.info(
        f"  Protocol  : N={N_TEST_USERS}, seed={SEED}, "
        f"MAX_BOOKS={MAX_BOOKS}, MAX_GENRES={MAX_GENRES}"
    )
    log.info(f"  K values  : {K_VALUES}")
    log.info("=" * 75)

    log.info("\nLoading data...")
    items_emb = np.load(cfg.items_emb_path).astype(np.float32)

    with open(cfg.items_idx_path) as f:
        item_id_to_idx: dict[str, int] = json.load(f)
    idx_to_item: dict[int, str] = {v: k for k, v in item_id_to_idx.items()}

    genre_gt = build_ground_truth(cfg, args.gt)
    test_users = sample_test_users(genre_gt, cfg)

    log.info(f"  Items      : {items_emb.shape}")
    log.info(f"  GT users   : {len(genre_gt):,}")
    log.info(f"  Test users : {len(test_users):,}")

    log.info("\nComputing random baseline...")
    random_metrics = compute_random_baseline(genre_gt, test_users, idx_to_item, len(idx_to_item))
    log.info(f"  Random: nDCG@10={random_metrics.get('nDCG@10', 0):.4f}")
    baselines = trivial_baselines(cfg, genre_gt, test_users, idx_to_item)

    if args.all:
        all_results = run_all(cfg, items_emb, idx_to_item, genre_gt, test_users, random_metrics)
        all_results[1:1] = baselines
    else:
        prof_path = Path(args.profiles) if args.profiles else PROCESSED / "user_profiles_h2.npy"
        label = args.label or prof_path.stem.replace("user_profiles_", "")
        log.info(f"\nEvaluating: {label}")

        if args.index:
            profiles = np.load(prof_path).astype(np.float32)
            with open(args.index) as f:
                idx_map: dict[str, int] = json.load(f)
        else:
            profiles, idx_map = load_profiles(prof_path)

        res = evaluate_method(
            label, profiles, items_emb, idx_map, idx_to_item, genre_gt, test_users
        )
        all_results = [
            {"label": "Random", **random_metrics},
            *baselines,
            {"label": label, **res},
        ]

    table = print_table(all_results)
    log.info(table)

    if not args.no_save:
        out_dir = RESULTS_DIR if args.gt == "genre" else RESULTS_DIR / f"{args.gt}_gt"
        out_dir.mkdir(parents=True, exist_ok=True)
        save_canonical_csv(all_results, ts, out_dir)
        summary_path = out_dir / f"summary_{cfg.domain}_{ts}.txt"
        with open(summary_path, "w") as f:
            f.write(f"Evaluation  [domain={cfg.domain}, gt={args.gt}]\n")
            f.write(f"Timestamp: {ts}\n")
            f.write(
                f"Protocol: N={N_TEST_USERS}, seed={SEED}, "
                f"MAX_BOOKS={MAX_BOOKS}, MAX_GENRES={MAX_GENRES}\n"
            )
            f.write(table)
        log.info(f"  Summary saved: {summary_path}")

    log.info("\nDONE")
