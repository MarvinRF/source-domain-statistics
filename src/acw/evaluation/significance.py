"""Paired significance testing for per-user metric vectors.

Claims are about means, so the primary test of the granularity experiments is
the paired sign-flip (permutation) test on the mean difference
(:func:`signflip_p`); the two-sided Wilcoxon signed-rank with Pratt's treatment
of zero differences is the secondary one. ``paired_test`` also returns the
number of users with a non-zero difference (``n_nonzero_diff``), the bootstrap
CI of the mean difference and the matched-pairs rank-biserial correlation
(Kerby's simple difference formula, signed so that r > 0 means ``a`` beats
``b``).
"""

from __future__ import annotations

import numpy as np
from scipy.stats import permutation_test, rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from acw.settings import SEED


def rank_biserial(a: np.ndarray, b: np.ndarray) -> float:
    """Signed matched-pairs rank-biserial r = (R+ - R-) / (R+ + R-), zero differences dropped."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 0.0
    ranks = rankdata(np.abs(d))
    return float((ranks[d > 0].sum() - ranks[d < 0].sum()) / ranks.sum())


def effect_label(r: float) -> str:
    r = abs(r)
    return "negligible" if r < 0.1 else "small" if r < 0.3 else "medium" if r < 0.5 else "large"


def bootstrap_ci(
    diff: np.ndarray, n_resamples: int = 10_000, level: float = 0.95
) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean paired difference."""
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(diff), size=(n_resamples, len(diff)))
    means = diff[idx].mean(axis=1)
    lo, hi = np.quantile(means, [(1 - level) / 2, (1 + level) / 2])
    return float(lo), float(hi)


def signflip_p(diff: np.ndarray, n_resamples: int = 10_000, max_cells: int = 8_000_000) -> float:
    """Two-sided Monte Carlo sign-flip (paired permutation) p-value of the mean of ``diff``.

    Under H0 each paired difference is equally likely to carry either sign, so
    the statistic ``|sum(s_i * d_i)|`` with random signs ``s_i`` is compared to
    the observed ``|sum(d_i)|``. Zero differences never change the statistic and
    are dropped, so the cost scales with the number of non-zero differences.
    The estimate is ``(1 + #{perm >= obs}) / (1 + n_resamples)`` (it cannot go
    below ``1 / (1 + n_resamples)``); the generator is seeded with ``SEED``.
    """
    d = np.asarray(diff, dtype=float)
    d = d[d != 0]
    if d.size == 0:
        return 1.0
    obs = abs(d.sum())
    rng = np.random.default_rng(SEED)
    chunk = max(1, min(n_resamples, max_cells // d.size))
    hits, done = 0, 0
    while done < n_resamples:
        size = min(chunk, n_resamples - done)
        signs = rng.integers(0, 2, size=(size, d.size), dtype=np.int8) * 2.0 - 1.0
        hits += int((np.abs(signs @ d) >= obs * (1 - 1e-12)).sum())
        done += size
    return (1 + hits) / (1 + n_resamples)


def paired_test(
    a: np.ndarray,
    b: np.ndarray,
    n_permutations: int = 10_000,
    n_bootstrap: int = 10_000,
    n_signflip: int = 0,
) -> dict:
    """Compare two per-user metric vectors aligned on the same users.

    ``n_permutations=0`` skips the scipy permutation test (``p_permutation`` is
    NaN); ``n_signflip > 0`` adds the sign-flip test ``p_signflip`` of
    :func:`signflip_p` (NaN otherwise); ``n_bootstrap`` sets the resamples of
    the CI of the mean difference.
    """
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    diff = a - b
    if np.any(diff):
        _, p_w = wilcoxon(a, b, alternative="two-sided", zero_method="pratt")
        if n_permutations > 0:
            p_perm = permutation_test(
                (a, b),
                lambda x, y: np.mean(x) - np.mean(y),
                permutation_type="samples",
                n_resamples=n_permutations,
                alternative="two-sided",
                random_state=SEED,
            ).pvalue
        else:
            p_perm = float("nan")
    else:
        p_w = 1.0
        p_perm = 1.0 if n_permutations > 0 else float("nan")
    r = rank_biserial(a, b)
    lo, hi = bootstrap_ci(diff, n_resamples=n_bootstrap)
    return {
        "n": int(len(a)),
        "n_nonzero_diff": int(np.count_nonzero(diff)),
        "mean_a": float(a.mean()),
        "mean_b": float(b.mean()),
        "mean_diff": float(diff.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_wilcoxon": float(p_w),
        "p_permutation": float(p_perm),
        "p_signflip": signflip_p(diff, n_signflip) if n_signflip > 0 else float("nan"),
        "rank_biserial_r": r,
        "effect": effect_label(r),
    }


def relative_gain_gap(
    raw_a: np.ndarray,
    cond_a: np.ndarray,
    raw_b: np.ndarray,
    cond_b: np.ndarray,
    n_resamples: int = 2_000,
    level: float = 0.95,
    chunk: int = 500,
) -> dict[str, np.ndarray]:
    """Difference of relative gains between two settings, with a paired bootstrap CI.

    ``raw_*`` are per-user metrics of the reference method, ``cond_*`` (n,) or
    (n, K) those of K methods, in setting ``a`` and ``b`` on the SAME users. The
    relative gain of a method in a setting is ``(mean(cond) - mean(raw)) /
    mean(raw)``; the gap is ``gain_b - gain_a``. Users are resampled jointly
    across settings and methods. Returns arrays of length K: ``gain_a``,
    ``gain_b``, ``gap``, ``ci95_low``, ``ci95_high``.
    """
    ra = np.asarray(raw_a, dtype=float)[:, None]
    rb = np.asarray(raw_b, dtype=float)[:, None]
    ca = np.asarray(cond_a, dtype=float).reshape(len(ra), -1)
    cb = np.asarray(cond_b, dtype=float).reshape(len(ra), -1)
    n, k = ca.shape
    values = np.hstack([ra, ca, rb, cb])

    def gains(means: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        m_ra, m_ca = means[..., :1], means[..., 1 : 1 + k]
        m_rb, m_cb = means[..., 1 + k : 2 + k], means[..., 2 + k :]
        return (m_ca - m_ra) / m_ra, (m_cb - m_rb) / m_rb

    gain_a, gain_b = gains(values.mean(axis=0))
    rng = np.random.default_rng(SEED)
    boot = []
    for start in range(0, n_resamples, chunk):
        size = min(chunk, n_resamples - start)
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=size)
        g_a, g_b = gains(counts @ values / n)
        boot.append(g_b - g_a)
    lo, hi = np.quantile(np.vstack(boot), [(1 - level) / 2, (1 + level) / 2], axis=0)
    return {
        "gain_a": gain_a,
        "gain_b": gain_b,
        "gap": gain_b - gain_a,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def holm(p_values: list[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Holm-Bonferroni adjusted p-values and rejection mask for one family of tests."""
    reject, p_adj, _, _ = multipletests(
        np.asarray(p_values, dtype=float), alpha=alpha, method="holm"
    )
    return p_adj, reject


def holm_rows(
    rows: list[dict], key_raw: str, key_holm: str, key_sig: str, alpha: float = 0.05
) -> None:
    """Holm-adjust ``row[key_raw]`` across ``rows`` (one family) in place."""
    p_adj, reject = holm([r[key_raw] for r in rows], alpha)
    for row, p, rej in zip(rows, p_adj, reject):
        row[key_holm] = round(float(p), 8)
        row[key_sig] = bool(rej)


def paired_row(
    cond: np.ndarray, ref: np.ndarray, n_bootstrap: int = 2_000, n_signflip: int = 10_000
) -> dict:
    """Columns reported for every paired comparison of ``cond`` against ``ref``
    (per-user vectors on the same users): the number of users ``n``, the means,
    ``mean_diff`` with its bootstrap 95% CI, ``n_nonzero_diff`` (users whose two
    values differ), the PRIMARY sign-flip p-value on the mean (``p_signflip``),
    the secondary Wilcoxon-Pratt p-value and the rank-biserial r (computed on
    the non-zero differences only)."""
    t = paired_test(cond, ref, 0, n_bootstrap, n_signflip)
    return {
        "n": t["n"],
        "mean_cond": t["mean_a"],
        "mean_ref": t["mean_b"],
        "mean_diff": t["mean_diff"],
        "ci95_low": t["ci95_low"],
        "ci95_high": t["ci95_high"],
        "n_nonzero_diff": t["n_nonzero_diff"],
        "p_signflip": t["p_signflip"],
        "p_wilcoxon": t["p_wilcoxon"],
        "rank_biserial_r": t["rank_biserial_r"],
    }


def add_holm_columns(rows: list[dict], family: list[dict] | None = None) -> None:
    """Add ``p_holm_signflip`` (primary) and ``p_holm_wilcoxon`` (secondary) to
    every row of ``rows``, Holm-adjusted within ``family`` (default: all of
    ``rows``; rows outside the family get NaN)."""
    members = rows if family is None else family
    for key, col in (("p_signflip", "p_holm_signflip"), ("p_wilcoxon", "p_holm_wilcoxon")):
        for r in rows:
            r[col] = float("nan")
        if members:
            p_adj, _ = holm([r[key] for r in members])
            for r, p in zip(members, p_adj):
                r[col] = float(p)
