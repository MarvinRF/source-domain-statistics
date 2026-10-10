"""
experiments/domains/games_zcabook_only.py
=========================================
Isolates the ZCA-book pre-stage on the games domain. When a learned
alignment module is preceded by ZCA-book (ZCA fitted on the 199,712 book
embeddings), the games-domain improvement is attributed to geometric
calibration rather than to the learned component. This script tests that
attribution directly by evaluating ZCA-book WITHOUT any learned module on
top, against the same catalog-whitening baseline (H2 vs. whitened catalog,
nDCG@10 = 0.4377). It reuses the evaluation data/functions that produced the
0.4377 result (src/acw/evaluation), retrains nothing and never overwrites an
existing result file.

Evaluated variant: H2 (raw, mean-pooled profile) -> apply_zca_book() (the
same book-fitted ZCA matrix used by the PW/MoE/NL-PW modules with the -ZB
suffix) -> cosine similarity against the SAME catalog-whitened games items
(games_embeddings_whitened.npy) -> nDCG@k / P@k.

Supports: the attribution of the learned-baseline gains on games to the
whitening pre-stage rather than to the learned module.

Inputs: data/processed/games/alignment/games_embeddings_whitened.npy,
        data/processed/user_profiles_h2_games.npy (+ index), games genre GT.
Output: results/phase4/zcabook_alone_games_evaluation.csv (a NEW file; the
        script refuses to run if it already exists).

Usage:
    python -m experiments.domains.games_zcabook_only
"""

from __future__ import annotations

import csv

import numpy as np

from acw.baselines.parametric_whitening import apply_zca_book  # noqa: E402
from acw.evaluation.config import K_VALUES, PROCESSED, resolve_paths  # noqa: E402
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users  # noqa: E402
from acw.evaluation.pipeline import evaluate_method  # noqa: E402
from acw.settings import ROOT  # noqa: E402


def main() -> None:
    cfg = resolve_paths(domain="games")
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)
    print(f"games: {len(test_users)} test users (after genre_gt filter)")

    # NOTE: cfg.items_emb_path/items_idx_path (resolve_paths defaults) point
    # to the RAW catalog (games_embeddings_qwen.npy), the generic domain
    # default used by the unwhitened "H2"/"H3"/"H4" rows in KNOWN_METHODS.
    # The "H2 (vs whitened)" row explicitly overrides items_emb_path with the
    # whitened catalog; that override is replicated here by loading the same
    # file referenced in acw/evaluation/config.py. The same index mapping
    # (games_id_to_idx_qwen.json) applies to both, since whitening preserves
    # row order and only transforms the values.
    items_emb = np.load(PROCESSED / "games" / "alignment" / "games_embeddings_whitened.npy").astype(
        np.float32
    )
    with open(cfg.items_idx_path) as f:
        import json

        item_id_to_idx: dict[str, int] = json.load(f)
    idx_to_item = {v: k for k, v in item_id_to_idx.items()}
    print(f"items_emb (catalog-whitened, games): {items_emb.shape}")

    h2_path = PROCESSED / "user_profiles_h2_games.npy"
    h2_idx_path = PROCESSED / "user_profile_index_h2_games.json"
    h2_profiles_raw = np.load(h2_path).astype(np.float32)
    with open(h2_idx_path) as f:
        import json

        h2_idx_map: dict[str, int] = json.load(f)
    print(f"H2 raw profiles (games): {h2_profiles_raw.shape}")

    # Reference baseline: H2 raw vs. whitened catalog (must reproduce the
    # reported nDCG@10 ~= 0.4377).
    baseline = evaluate_method(
        "H2 (vs whitened) [reproduced baseline]",
        h2_profiles_raw,
        items_emb,
        h2_idx_map,
        idx_to_item,
        genre_gt,
        test_users,
    )

    # New variant: H2 + ZCA-book (no learned module), same whitened catalog
    # and same ground truth as the baseline above.
    h2_zca_book = apply_zca_book(h2_profiles_raw)
    print(f"H2 + ZCA-book (games): {h2_zca_book.shape}")
    zcabook_alone = evaluate_method(
        "H2 + ZCA-book (no learned module)",
        h2_zca_book,
        items_emb,
        h2_idx_map,
        idx_to_item,
        genre_gt,
        test_users,
    )

    out_path = ROOT / "results" / "phase4" / "zcabook_alone_games_evaluation.csv"
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists; delete it manually before re-running "
            "to avoid accidentally overwriting a previous result."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "n_eval",
        *[f"nDCG@{k}" for k in K_VALUES],
        *[f"P@{k}" for k in K_VALUES],
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for label, metrics in [
            ("h2_vs_whitened_baseline_reproduced", baseline),
            ("h2_zca_book_alone_no_learned_module", zcabook_alone),
        ]:
            row = {"method": label, "n_eval": metrics["_n_eval"]}
            row.update({k: metrics[k] for k in fieldnames if k not in ("method", "n_eval")})
            writer.writerow(row)
    print(f"\nSaved to {out_path}")

    print("\n=== SUMMARY ===")
    print(
        f"H2 vs whitened (reproduced baseline):  nDCG@10={baseline['nDCG@10']:.4f} (n={baseline['_n_eval']})"
    )
    print(
        f"H2 + ZCA-book, no learned module:      nDCG@10={zcabook_alone['nDCG@10']:.4f} (n={zcabook_alone['_n_eval']})"
    )
    delta = zcabook_alone["nDCG@10"] - baseline["nDCG@10"]
    pct = 100 * delta / baseline["nDCG@10"]
    print(f"Delta: {delta:+.4f} ({pct:+.1f}%)")


if __name__ == "__main__":
    main()
