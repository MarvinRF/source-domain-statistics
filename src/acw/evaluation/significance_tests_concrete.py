"""Significance tests for the closed-vocabulary ("concrete") prompt variants
(H3-concrete, H4-concrete) against the free-form evocative profiles (H3, H4)
and against the global baseline (H2+albuns-whitened), music domain.

Family (k=4): Wilcoxon(pratt) + permutation test (n=10,000, seed=42) + Holm.
  C1: H3-concrete vs H3 — does a closed vocabulary help or hurt relative to
      the free-form prompt within the same profile family?
  C2: H4-concrete vs H4.
  C3: H3-concrete vs H2+albuns-whitened (global baseline).
  C4: H4-concrete vs H2+albuns-whitened.

Pairs are aligned on the intersection of user_ids (not by position).

Usage
-----
    python -m acw.evaluation.significance_tests_concrete
"""

from __future__ import annotations

import csv
import json
import logging
import sys

import numpy as np
from scipy.stats import permutation_test, wilcoxon

from acw.evaluation.config import KNOWN_METHODS, PROCESSED, resolve_paths
from acw.evaluation.loader import build_genre_ground_truth, load_profiles, sample_test_users
from acw.evaluation.metrics import ndcg_at_k
from acw.evaluation.significance import effect_label, holm_rows, rank_biserial
from acw.settings import ROOT  # noqa: E402

ST_DIR = ROOT / "results" / "statistical_tests"
ST_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SEED = 42
N_PERMUTATIONS = 10000
DOMAIN = "music"


def compute_per_user_ndcg10(
    label: str,
    genre_gt: dict[str, set[str]],
    test_users: list[str],
    idx_to_item: dict[int, str],
    default_items_emb: np.ndarray,
) -> dict[str, float]:
    entry = next((e for e in KNOWN_METHODS.get(DOMAIN, []) if e[0] == label), None)
    if entry is None:
        raise ValueError(f"Method {label!r} is not registered for domain={DOMAIN}")
    _, rel_path, idx_rel, emb_rel, _ = entry
    path = PROCESSED / rel_path
    profiles = np.load(path).astype(np.float32)
    if idx_rel:
        with open(PROCESSED / idx_rel) as f:
            idx_map: dict[str, int] = json.load(f)
    else:
        _, idx_map = load_profiles(path)
    items_emb = np.load(PROCESSED / emb_rel).astype(np.float32) if emb_rel else default_items_emb
    out: dict[str, float] = {}
    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        row = idx_map.get(uid)
        if row is None:
            continue
        sims = items_emb @ profiles[row]
        top = np.argsort(sims)[::-1][:20]
        ranked = [idx_to_item[i] for i in top]
        out[uid] = ndcg_at_k(ranked, relevant, 10)
    return out


def run_pair(
    pair_id: str,
    method_a: str,
    method_b: str,
    data_a: dict[str, float],
    data_b: dict[str, float],
) -> dict:
    common = sorted(set(data_a) & set(data_b))
    n = len(common)
    a = np.array([data_a[u] for u in common])
    b = np.array([data_b[u] for u in common])
    mean_a, mean_b = float(np.mean(a)), float(np.mean(b))
    try:
        _, p_w = wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
    except ValueError:
        p_w = 1.0

    def _stat(x, y):
        return np.mean(x) - np.mean(y)

    perm = permutation_test(
        (a, b),
        _stat,
        permutation_type="samples",
        n_resamples=N_PERMUTATIONS,
        alternative="two-sided",
        random_state=SEED,
    )
    r = rank_biserial(a, b)
    return {
        "pair_id": pair_id,
        "method_a": method_a,
        "method_b": method_b,
        "n_common": n,
        "mean_ndcg10_a": round(mean_a, 4),
        "mean_ndcg10_b": round(mean_b, 4),
        "delta_ndcg10": round(mean_a - mean_b, 4),
        "p_wilcoxon_raw": p_w,
        "p_wilcoxon_holm": None,
        "p_perm_raw": float(perm.pvalue),
        "p_perm_holm": None,
        "effect_size_r": round(r, 4),
        "effect_size_label": effect_label(r),
        "significant_wilcoxon_holm": None,
        "significant_perm_holm": None,
    }


def main() -> None:
    log.info("=" * 70)
    log.info("Concrete vs evocative vs H2-whitened (k=4, Holm)")
    log.info("=" * 70)

    cfg = resolve_paths(domain=DOMAIN)
    default_items_emb = np.load(cfg.items_emb_path).astype(np.float32)
    with open(cfg.items_idx_path) as f:
        item_id_to_idx: dict[str, int] = json.load(f)
    idx_to_item = {v: k for k, v in item_id_to_idx.items()}
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)
    log.info(f"  Test users (canonical val): {len(test_users):,}")

    labels = ["H3-concrete", "H4-concrete", "H3", "H4", "H2+albuns-whitened"]
    per_user: dict[str, dict[str, float]] = {}
    for label in labels:
        data = compute_per_user_ndcg10(label, genre_gt, test_users, idx_to_item, default_items_emb)
        per_user[label] = data
        mean = np.mean(list(data.values())) if data else float("nan")
        log.info(f"  {label:<30}: n={len(data)}  nDCG@10={mean:.4f}")
        out_path = ST_DIR / f"per_user_concrete_{label.replace(' ', '_').replace('+', 'plus')}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["user_id", "ndcg10"])
            for uid, v in data.items():
                writer.writerow([uid, v])

    pairs = [
        ("C1", "H3-concrete", "H3", "concrete vs evocative (H3 family)"),
        ("C2", "H4-concrete", "H4", "concrete vs evocative (H4 family)"),
        ("C3", "H3-concrete", "H2+albuns-whitened", "H3-concrete vs baseline global"),
        ("C4", "H4-concrete", "H2+albuns-whitened", "H4-concrete vs baseline global"),
    ]
    rows = []
    for pair_id, ma, mb, _desc in pairs:
        row = run_pair(pair_id, ma, mb, per_user[ma], per_user[mb])
        log.info(
            f"  {pair_id} {ma:<18} vs {mb:<20}: "
            f"Δ={row['delta_ndcg10']:+.4f}  n={row['n_common']}  "
            f"p_w={row['p_wilcoxon_raw']:.2e}  p_perm={row['p_perm_raw']:.2e}  "
            f"r={row['effect_size_r']:.3f} ({row['effect_size_label']})"
        )
        rows.append(row)

    holm_rows(rows, "p_wilcoxon_raw", "p_wilcoxon_holm", "significant_wilcoxon_holm")
    holm_rows(rows, "p_perm_raw", "p_perm_holm", "significant_perm_holm")

    log.info(f"\nHolm-adjusted results (k={len(rows)}):")
    for row in rows:
        sig_w = "✓" if row["significant_wilcoxon_holm"] else "✗"
        sig_p = "✓" if row["significant_perm_holm"] else "✗"
        log.info(
            f"  {row['pair_id']} {row['method_a']:<18} vs {row['method_b']:<20} "
            f"Δ={row['delta_ndcg10']:+.4f} "
            f"p_holm_w={row['p_wilcoxon_holm']:.4f}{sig_w} "
            f"p_holm_perm={row['p_perm_holm']:.4f}{sig_p} "
            f"r={row['effect_size_r']:.3f}"
        )

    out_path = ST_DIR / "significance_results_concrete.csv"
    fieldnames = [
        "pair_id",
        "method_a",
        "method_b",
        "n_common",
        "mean_ndcg10_a",
        "mean_ndcg10_b",
        "delta_ndcg10",
        "p_wilcoxon_raw",
        "p_wilcoxon_holm",
        "p_perm_raw",
        "p_perm_holm",
        "effect_size_r",
        "effect_size_label",
        "significant_wilcoxon_holm",
        "significant_perm_holm",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    log.info(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
