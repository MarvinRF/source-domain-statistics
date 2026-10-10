"""Per-user metrics on the canonical evaluation protocol (music domain).

Writes one row per evaluated user so that paired significance tests
(Wilcoxon, permutation test, effect size) can be run on the distributions.
Uses the same protocol as ``acw.evaluation.evaluate``; each method is scored
against its matching item embeddings (raw or whitened).

Usage
-----
    # All predefined methods
    python -m acw.evaluation.per_user_metrics --all

    # Single method
    python -m acw.evaluation.per_user_metrics \\
        --profiles data/processed/user_profiles_h5_alpha090.npy \\
        --emb data/processed/alignment/album_embeddings_whitened.npy \\
        --label H5-alpha090

Output
------
    results/statistical_tests/per_user_{label}.csv
    Columns: user_id, ndcg5, ndcg10, ndcg20, p10
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np

from acw.evaluation.config import K_VALUES, SEED, resolve_paths
from acw.evaluation.loader import build_ground_truth, sample_test_users
from acw.evaluation.metrics import ndcg_at_k, precision_at_k
from acw.evaluation.ranking import top_k
from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ALIGNMENT = PROCESSED / "alignment"
OUT_DIR = ROOT / "results" / "statistical_tests"
OUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─── Optional expected nDCG@10 per label for sanity checks (tolerance ±0.001) ─
# None = no reference value; the mean is only logged.

_CANONICAL_NDCG10: dict[str, float | None] = {
    "H5-alpha095": None,
    "H2": None,
    "H3-PW-ZB": None,
    "H3-sg-pw-zb": None,
    "H3-PW-Linear": None,
    "H4-PW-Linear": None,
    "H3-ZCA-profile": None,
    "Random": None,
}

# ─── Method definitions (profiles, idx, items_emb) ───────────────────────────
# Item embeddings per method:
#   raw (album_embeddings.npy)               → H2, Random
#   whitened (album_embeddings_whitened.npy) → all aligned methods

_EMB_RAW = PROCESSED / "album_embeddings.npy"
_EMB_WHITENED = ALIGNMENT / "album_embeddings_whitened.npy"

_METHODS: list[dict] = [
    {
        "label": "H5-alpha095",
        "profiles": PROCESSED / "user_profiles_h5_alpha095.npy",
        "idx": PROCESSED / "user_profile_index_h5_alpha095.json",
        "emb": _EMB_WHITENED,
    },
    {
        "label": "H2",
        "profiles": PROCESSED / "user_profiles_h2.npy",
        "idx": PROCESSED / "user_profile_index_h2.json",
        "emb": _EMB_RAW,
    },
    {
        "label": "H3-PW-ZB",
        "profiles": ALIGNMENT / "user_profiles_h3_pw_zb.npy",
        "idx": PROCESSED / "user_profile_index_h3.json",
        "emb": _EMB_WHITENED,
    },
    {
        "label": "H3-sg-pw-zb",
        "profiles": ALIGNMENT / "user_profiles_h3_sg_pw_zb.npy",
        "idx": ALIGNMENT / "user_profile_index_h3_sg_pw_zb.json",
        "emb": _EMB_WHITENED,
    },
    {
        "label": "H3-PW-Linear",
        "profiles": ALIGNMENT / "user_profiles_h3_pw.npy",
        "idx": PROCESSED / "user_profile_index_h3.json",
        "emb": _EMB_WHITENED,
    },
    {
        "label": "H4-PW-Linear",
        "profiles": ALIGNMENT / "user_profiles_h4_pw.npy",
        "idx": PROCESSED / "user_profile_index_h4.json",
        "emb": _EMB_WHITENED,
    },
    {
        "label": "H3-ZCA-profile",
        "profiles": ALIGNMENT / "user_profiles_h3_profile_whitened.npy",
        "idx": PROCESSED / "user_profile_index_h3.json",
        "emb": _EMB_WHITENED,
    },
    {
        "label": "Random",
        "profiles": None,
        "idx": PROCESSED / "user_profile_index_h2.json",
        "emb": _EMB_RAW,
    },
]


# ─── Per-user computation ─────────────────────────────────────────────────────


def compute_per_user(
    label: str,
    profiles: np.ndarray | None,
    idx_map: dict[str, int],
    items_emb: np.ndarray,
    genre_gt: dict[str, set[str]],
    test_users: list[str],
    idx_to_item: dict[int, str],
) -> list[dict]:
    max_k = max(K_VALUES)
    rows: list[dict] = []

    if profiles is None:
        # Random baseline — mirrors pipeline.compute_random_baseline exactly
        rng = np.random.default_rng(SEED)
        n_items = items_emb.shape[0]
        for uid in test_users:
            relevant = genre_gt.get(uid)
            if not relevant:
                continue
            idxs = rng.choice(n_items, size=max_k, replace=False)
            ranked = [idx_to_item[i] for i in idxs]
            rows.append(
                {
                    "user_id": uid,
                    "ndcg5": ndcg_at_k(ranked, relevant, 5),
                    "ndcg10": ndcg_at_k(ranked, relevant, 10),
                    "ndcg20": ndcg_at_k(ranked, relevant, 20),
                    "p10": precision_at_k(ranked, relevant, 10),
                }
            )
        return rows

    users = [u for u in test_users if genre_gt.get(u) and idx_map.get(u) is not None]
    tops = top_k(profiles[[idx_map[u] for u in users]], items_emb, max_k)
    for uid, top in zip(users, tops):
        relevant = genre_gt[uid]
        ranked = [idx_to_item[i] for i in top]
        rows.append(
            {
                "user_id": uid,
                "ndcg5": ndcg_at_k(ranked, relevant, 5),
                "ndcg10": ndcg_at_k(ranked, relevant, 10),
                "ndcg20": ndcg_at_k(ranked, relevant, 20),
                "p10": precision_at_k(ranked, relevant, 10),
            }
        )

    return rows


def sanity_check(label: str, rows: list[dict]) -> bool:
    if not rows:
        log.warning(f"  {label}: empty CSV")
        return False

    mean10 = float(np.mean([r["ndcg10"] for r in rows]))
    expected = _CANONICAL_NDCG10.get(label)

    if expected is None:
        log.info(f"  {label}: nDCG@10={mean10:.4f}  (no reference value)")
        return True

    diff = abs(mean10 - expected)
    ok = diff <= 0.001
    flag = "✓" if ok else "✗ ERROR"
    log.info(
        f"  {label}: mean nDCG@10={mean10:.4f}  (expected: {expected:.4f}, diff: {diff:.4f}) {flag}"
    )
    if not ok:
        log.error(f"  {label}: sanity check FAILED — check items_emb and pipeline")
    return ok


def run_one(
    label: str,
    npy_path: Path | None,
    idx_path: Path,
    emb_path: Path,
    genre_gt: dict[str, set[str]],
    test_users: list[str],
    idx_to_item: dict[int, str],
    output: Path | None = None,
) -> list[dict]:
    log.info(f"\n{'─' * 60}")
    log.info(f"  Method : {label}")
    log.info(f"  items_emb: {emb_path.name}")

    if npy_path is not None:
        if not npy_path.exists():
            log.warning(f"  {label}: {npy_path} not found — skipping")
            return []
        profiles_arr: np.ndarray | None = np.load(npy_path).astype(np.float32)
    else:
        profiles_arr = None

    if not idx_path.exists():
        log.warning(f"  {label}: idx {idx_path} not found — skipping")
        return []
    with open(idx_path) as f:
        idx_map: dict[str, int] = json.load(f)

    items_emb = np.load(emb_path).astype(np.float32)

    rows = compute_per_user(
        label, profiles_arr, idx_map, items_emb, genre_gt, test_users, idx_to_item
    )

    sanity_check(label, rows)

    out_path = output or (OUT_DIR / f"per_user_{label}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "ndcg5", "ndcg10", "ndcg20", "p10"])
        writer.writeheader()
        writer.writerows(rows)

    log.info(f"  Saved: {out_path}  ({len(rows)} users)")
    return rows


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-user metrics for significance testing"
    )
    parser.add_argument("--profiles", type=Path, default=None)
    parser.add_argument(
        "--emb", type=Path, default=None, help="Path to the album embeddings (.npy)"
    )
    parser.add_argument("--label", type=str, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--all", action="store_true", help="Run all 8 predefined methods")
    parser.add_argument(
        "--gt",
        choices=("exact", "genre"),
        default="genre",
        help="exact = items actually consumed; genre = subgenre expansion.",
    )
    args = parser.parse_args()

    global OUT_DIR
    if args.gt != "genre":
        # Reference sanity values apply only to the subgenre ground truth
        OUT_DIR = OUT_DIR / f"{args.gt}_gt"
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        _CANONICAL_NDCG10.clear()

    if not args.all and (args.profiles is None or args.label is None or args.emb is None):
        parser.error("--profiles, --emb and --label are required without --all")

    log.info("=" * 70)
    log.info("Per-user metrics (significance tests)")
    log.info("=" * 70)

    cfg = resolve_paths("music")
    genre_gt = build_ground_truth(cfg, args.gt)
    test_users = sample_test_users(genre_gt, cfg)

    with open(cfg.items_idx_path) as f:
        item_id_to_idx: dict[str, int] = json.load(f)
    idx_to_item = {v: k for k, v in item_id_to_idx.items()}

    log.info(f"  test_users: {len(test_users)}  |  idx_to_item: {len(idx_to_item)}")

    all_ok = True

    if args.all:
        for m in _METHODS:
            rows = run_one(
                label=m["label"],
                npy_path=m["profiles"],
                idx_path=m["idx"],
                emb_path=m["emb"],
                genre_gt=genre_gt,
                test_users=test_users,
                idx_to_item=idx_to_item,
            )
            if rows:
                mean10 = np.mean([r["ndcg10"] for r in rows])
                expected = _CANONICAL_NDCG10.get(m["label"])
                if expected is not None and abs(mean10 - expected) > 0.001:
                    all_ok = False
    else:
        run_one(
            label=args.label,
            npy_path=args.profiles,
            idx_path=args.profiles.with_name(
                args.profiles.stem.replace("user_profiles_", "user_profile_index_") + ".json"
            ),
            emb_path=args.emb,
            genre_gt=genre_gt,
            test_users=test_users,
            idx_to_item=idx_to_item,
            output=args.output,
        )

    log.info("\n" + "=" * 70)
    if all_ok:
        log.info("DONE — all sanity checks passed ✓")
    else:
        log.error("DONE — sanity check FAILED ✗ — investigate before proceeding")
    log.info("=" * 70)

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
