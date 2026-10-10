"""
Size-matched resampling for the query-side register-gap test.

The query-side test compares two term groups of very different size
(group A, n=50, vs. group B, n=5,138). To obtain a clean two-condition
comparison, this script repeatedly draws size-matched subsamples of B
(n = |A| = 50, without replacement), recomputes the test on each draw, and
reports the DISTRIBUTION of the gap (%), the absolute rank-biserial effect
size and the fraction of significant draws, rather than a single point.

Supports: the robustness of the query-side register gap to group-size
imbalance.

Input:  results/phase4/dla_term_level_similarities.csv (the same CSV as the
        original test; no re-embedding is needed).
Output: results/statistical_tests/dla_bootstrap_resample_n50.json

Usage:
    python -m experiments.register_gap.span_bootstrap
"""

import csv
import json

import numpy as np
from scipy.stats import mannwhitneyu

from acw.settings import ROOT  # noqa: E402

SRC_CSV = ROOT / "results" / "phase4" / "dla_term_level_similarities.csv"
OUT_JSON = ROOT / "results" / "statistical_tests" / "dla_bootstrap_resample_n50.json"

SEED = 42
N_RESAMPLES = 5000


def main():
    rows = list(csv.DictReader(open(SRC_CSV)))
    a = np.array([float(r["mean_similarity_to_albums"]) for r in rows if r["group"] == "A"])
    b_full = np.array([float(r["mean_similarity_to_albums"]) for r in rows if r["group"] == "B"])
    n_match = len(a)
    print(f"n_A = {len(a)}, n_B_full = {len(b_full)}, resample size = {n_match}")

    rng = np.random.default_rng(SEED)
    gaps, rs, pvals = [], [], []
    for _ in range(N_RESAMPLES):
        sub = rng.choice(b_full, size=n_match, replace=False)
        gap = (a.mean() - sub.mean()) / a.mean() * 100
        U, p = mannwhitneyu(a, sub, alternative="two-sided")
        r = 1 - (2 * U) / (len(a) * len(sub))  # rank-biserial from Mann-Whitney U
        gaps.append(gap)
        rs.append(abs(r))
        pvals.append(p)

    gaps, rs, pvals = np.array(gaps), np.array(rs), np.array(pvals)
    result = {
        "seed": SEED,
        "n_resamples": N_RESAMPLES,
        "n_per_group": int(n_match),
        "gap_pct": {
            "mean": float(gaps.mean()),
            "median": float(np.median(gaps)),
            "sd": float(gaps.std()),
            "ci95": [float(np.percentile(gaps, 2.5)), float(np.percentile(gaps, 97.5))],
        },
        "abs_rank_biserial_r": {
            "mean": float(rs.mean()),
            "median": float(np.median(rs)),
            "sd": float(rs.std()),
            "ci95": [float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))],
        },
        "frac_p_lt_05": float((pvals < 0.05).mean()),
        "frac_same_direction_A_gt_B": float((gaps > 0).mean()),
        "original_reported": {"gap_pct": 3.8, "r": 0.16, "p_permutation": 0.011},
    }
    print(json.dumps(result, indent=2))

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
