"""
Formal interaction test: profile family (H3 vs H4) × ZCA-book pre-stage (ZB vs no-ZB)
on the games domain per-user nDCG@10 scores.

Tests whether the opposite direction of the ZB effect for H3 and H4 in games is
statistically reliable.

Protocol:
  - For each alignment module M in {PW, MoE-PW}:
    - Intersect users with valid per-user scores across all four conditions
      (H3+M, H3+M-ZB, H4+M, H4+M-ZB).
    - Compute ZB effect for H3: delta_H3 = H3+M-ZB − H3+M (per user)
    - Compute ZB effect for H4: delta_H4 = H4+M-ZB − H4+M (per user)
    - Test H0: delta_H4 == delta_H3 via Wilcoxon signed-rank (Pratt) +
      permutation test (n=10,000, seed=42).

Usage (from the repository root):
    python -m acw.evaluation.interaction_test_games

Output: prints results and saves to results/statistical_tests/interaction_test_games.json
"""

import json
import os

import numpy as np
import pandas as pd
from scipy import stats

from acw.evaluation.significance import rank_biserial

BASE = "results/statistical_tests"


def load_ndcg10(fname: str) -> pd.Series:
    df = pd.read_csv(os.path.join(BASE, fname))
    user_col = df.columns[0]
    ndcg_col = [c for c in df.columns if "10" in c.lower()][0]
    return df.set_index(user_col)[ndcg_col]


def run_interaction(module: str) -> dict:
    h3 = load_ndcg10(f"per_user_games_H3+{module}.csv")
    h3_zb = load_ndcg10(f"per_user_games_H3+{module}-ZB.csv")
    h4 = load_ndcg10(f"per_user_games_H4+{module}.csv")
    h4_zb = load_ndcg10(f"per_user_games_H4+{module}-ZB.csv")

    common = h3.index.intersection(h3_zb.index).intersection(h4.index).intersection(h4_zb.index)
    n = len(common)

    d_h3 = (h3_zb - h3).loc[common].values
    d_h4 = (h4_zb - h4).loc[common].values

    stat_w, p_w = stats.wilcoxon(d_h4, d_h3, zero_method="pratt")
    r_rb = rank_biserial(d_h4, d_h3)

    def statistic_fn(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    perm = stats.permutation_test(
        (d_h4, d_h3),
        statistic_fn,
        n_resamples=10_000,
        random_state=42,
        alternative="two-sided",
    )

    return {
        "module": module,
        "n_users_common": n,
        "h3_zb_effect_mean": round(float(d_h3.mean()), 4),
        "h4_zb_effect_mean": round(float(d_h4.mean()), 4),
        "interaction_mean": round(float((d_h4 - d_h3).mean()), 4),
        "wilcoxon_stat": float(stat_w),
        "p_wilcoxon": float(p_w),
        "p_permutation": float(perm.pvalue),
        "r_rank_biserial": round(r_rb, 3),
        "significant_both": bool(p_w < 0.05 and perm.pvalue < 0.05),
    }


def run_interaction_moe() -> dict:
    """MoE uses MoE-PW (no ZB) vs MoE-ZB (with ZB) — different naming convention."""
    h3 = load_ndcg10("per_user_games_H3+MoE-PW.csv")
    h3_zb = load_ndcg10("per_user_games_H3+MoE-ZB.csv")
    h4 = load_ndcg10("per_user_games_H4+MoE-PW.csv")
    h4_zb = load_ndcg10("per_user_games_H4+MoE-ZB.csv")

    common = h3.index.intersection(h3_zb.index).intersection(h4.index).intersection(h4_zb.index)
    n = len(common)

    d_h3 = (h3_zb - h3).loc[common].values
    d_h4 = (h4_zb - h4).loc[common].values

    stat_w, p_w = stats.wilcoxon(d_h4, d_h3, zero_method="pratt")
    r_rb = rank_biserial(d_h4, d_h3)

    def statistic_fn(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    perm = stats.permutation_test(
        (d_h4, d_h3),
        statistic_fn,
        n_resamples=10_000,
        random_state=42,
        alternative="two-sided",
    )

    return {
        "module": "MoE",
        "n_users_common": n,
        "h3_zb_effect_mean": round(float(d_h3.mean()), 4),
        "h4_zb_effect_mean": round(float(d_h4.mean()), 4),
        "interaction_mean": round(float((d_h4 - d_h3).mean()), 4),
        "wilcoxon_stat": float(stat_w),
        "p_wilcoxon": float(p_w),
        "p_permutation": float(perm.pvalue),
        "r_rank_biserial": round(r_rb, 3),
        "significant_both": bool(p_w < 0.05 and perm.pvalue < 0.05),
    }


def main():
    results = {}
    for module in ["PW"]:
        r = run_interaction(module)
        results[module] = r
        print(
            f"{module} (n={r['n_users_common']}): "
            f"ΔH3={r['h3_zb_effect_mean']:+.4f}, ΔH4={r['h4_zb_effect_mean']:+.4f}, "
            f"interaction={r['interaction_mean']:+.4f}, "
            f"p_wilcoxon={r['p_wilcoxon']:.2e}, p_perm={r['p_permutation']:.4f}, "
            f"r={r['r_rank_biserial']:.3f}"
        )

    r_moe = run_interaction_moe()
    results["MoE"] = r_moe
    print(
        f"MoE (n={r_moe['n_users_common']}): "
        f"ΔH3={r_moe['h3_zb_effect_mean']:+.4f}, ΔH4={r_moe['h4_zb_effect_mean']:+.4f}, "
        f"interaction={r_moe['interaction_mean']:+.4f}, "
        f"p_wilcoxon={r_moe['p_wilcoxon']:.2e}, p_perm={r_moe['p_permutation']:.4f}, "
        f"r={r_moe['r_rank_biserial']:.3f}"
    )

    out = os.path.join(BASE, "interaction_test_games.json")
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
