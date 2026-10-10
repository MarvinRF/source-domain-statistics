import numpy as np
import pytest

from acw.evaluation.significance import holm, paired_test, rank_biserial, relative_gain_gap


def test_rank_biserial_sign_and_bounds():
    a = np.array([0.5, 0.6, 0.7, 0.8])
    b = a - 0.1
    assert rank_biserial(a, b) == 1.0
    assert rank_biserial(b, a) == -1.0
    assert rank_biserial(a, a) == 0.0


def test_rank_biserial_ignores_ties():
    # 1 win (small), 1 loss (large), many zero differences: r must stay negative
    a = np.array([0.2, 0.0] + [0.3] * 50)
    b = np.array([0.1, 0.5] + [0.3] * 50)
    assert rank_biserial(a, b) < 0


def test_paired_test_identical_vectors():
    x = np.linspace(0, 1, 20)
    out = paired_test(x, x, n_permutations=100)
    assert out["p_wilcoxon"] == 1.0 and out["mean_diff"] == 0.0


def test_paired_test_detects_shift():
    rng = np.random.default_rng(1)
    b = rng.uniform(size=200)
    out = paired_test(b + 0.1, b, n_permutations=200)
    assert out["p_wilcoxon"] < 1e-10 and out["ci95_low"] > 0 and out["rank_biserial_r"] == 1.0


def test_holm_is_monotone_and_conservative():
    p_adj, reject = holm([0.01, 0.04, 0.03])
    assert all(p_adj >= np.array([0.01, 0.04, 0.03]))
    assert reject.tolist() == [True, False, False]


def test_paired_test_without_permutation_or_with_fewer_bootstraps():
    rng = np.random.default_rng(0)
    a = rng.random(300)
    out = paired_test(a + 0.1, a, n_permutations=0, n_bootstrap=200)
    assert np.isnan(out["p_permutation"]) and out["p_wilcoxon"] < 1e-6
    assert out["ci95_low"] > 0 and out["mean_diff"] == pytest.approx(0.1)


def test_relative_gain_gap_detects_a_difference_in_gains():
    rng = np.random.default_rng(1)
    n = 2_000
    raw_a, raw_b = rng.random(n) + 0.5, rng.random(n) + 0.5
    # method 0: same +50% gain in both settings; method 1: +50% in a, +100% in b
    cond_a = np.column_stack([raw_a * 1.5, raw_a * 1.5])
    cond_b = np.column_stack([raw_b * 1.5, raw_b * 2.0])
    out = relative_gain_gap(raw_a, cond_a, raw_b, cond_b, n_resamples=300)
    np.testing.assert_allclose(out["gap"], [0.0, 0.5], atol=1e-12)
    assert abs(out["ci95_low"][0]) < 1e-9 and abs(out["ci95_high"][0]) < 1e-9
    assert out["ci95_low"][1] > 0.3 and out["ci95_high"][1] < 0.7


def test_signflip_p_detects_a_shift_ignores_zeros_and_is_seeded():
    from acw.evaluation.significance import signflip_p

    rng = np.random.default_rng(3)
    d = np.concatenate([rng.normal(0.3, 0.2, 80), np.zeros(500)])
    assert signflip_p(d, 2_000) < 0.01
    assert signflip_p(d, 2_000) == signflip_p(d[:80], 2_000)  # zeros never matter
    assert signflip_p(np.zeros(10)) == 1.0
    assert signflip_p(rng.normal(0, 1, 200), 2_000) > 0.01  # centred noise


def test_paired_row_reports_the_discordant_count_and_holm_columns():
    from acw.evaluation.significance import add_holm_columns, paired_row

    rng = np.random.default_rng(1)
    b = np.zeros(300)
    a = b.copy()
    a[:40] = rng.uniform(0.1, 0.5, 40)
    r = paired_row(a, b, n_bootstrap=200, n_signflip=2_000)
    assert r["n_nonzero_diff"] == 40 and r["n"] == 300
    assert r["p_signflip"] < 0.001 and r["ci95_low"] > 0
    rows = [dict(r), dict(r, p_signflip=0.5, p_wilcoxon=0.5)]
    add_holm_columns(rows)
    assert rows[0]["p_holm_signflip"] <= rows[1]["p_holm_signflip"] <= 1.0
