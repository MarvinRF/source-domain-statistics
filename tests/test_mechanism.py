import numpy as np

from experiments.granularity.mechanism import (
    between_within,
    category_concentration,
    geometry,
    gini,
    jaccard_to_list,
    spearman,
)


def _unit(n, d, seed=0):
    x = np.random.default_rng(seed).normal(size=(n, d))
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_mean_pairwise_cosine_closed_form():
    x = _unit(60, 8)
    s = x @ x.T
    brute = (s.sum() - np.trace(s)) / (60 * 59)
    assert abs(geometry(x)["mean_pairwise_cosine"] - brute) < 1e-6


def test_effective_rank_bounds():
    iso = geometry(_unit(5000, 6))
    assert 5.9 < iso["effective_rank"] <= 6.0
    sign = np.sign(np.random.default_rng(2).normal(size=(200, 1)))
    g = geometry((sign * np.eye(6)[0]).astype(np.float32))
    assert abs(g["effective_rank"] - 1) < 1e-3 and abs(g["var_share_pc1"] - 1) < 1e-6


def test_gini():
    assert abs(gini(np.ones(10))) < 1e-12
    assert abs(gini(np.array([0, 0, 0, 10])) - 0.75) < 1e-12


def test_between_within_separates_clusters():
    rng = np.random.default_rng(1)
    codes = np.repeat([0, 1], 100)
    x = rng.normal(scale=0.1, size=(200, 4)) + 5.0 * codes[:, None] * np.eye(4)[0]
    tight = between_within(x.astype(np.float32), codes)
    assert tight > 100 and between_within(x.astype(np.float32), rng.permutation(codes)) < 0.1


def test_category_concentration():
    cat = np.array([0, 0, 1, 1, 2, 2, 3, 3])  # uniform over 4 categories
    out = category_concentration(cat, [0, 1, 2, 3, 3], 4)
    assert abs(out["entropy_bits"] - 2.0) < 1e-12 and out["n_distinct_categories"] == 4
    assert out["share_top1_train_category"] == 0.25 and out["share_top5_train_categories"] == 1.0
    one = category_concentration(np.array([2, 2, -1]), [2], 3)
    assert one["entropy_bits"] == 0 and one["share_uncategorised"] == 1 / 3


def test_jaccard_to_list_and_spearman():
    top = np.array([[0, 1, 2], [3, 4, 5], [0, 1, 9]])
    assert np.allclose(jaccard_to_list(top, np.array([0, 1, 2])), [1.0, 0.0, 0.5])
    assert spearman(np.ones(5), np.arange(5)) == {"rho": None, "p": None}
    assert abs(spearman(np.arange(5), np.arange(5) ** 3)["rho"] - 1) < 1e-12
