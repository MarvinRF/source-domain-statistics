import math

import numpy as np

from acw.evaluation.metrics import ndcg_at_k, precision_at_k, recall_at_k
from acw.evaluation.ranking import top_k


def test_ndcg_perfect_and_empty():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}, 3) == 1.0
    assert ndcg_at_k(["x", "y"], {"a"}, 2) == 0.0


def test_ndcg_known_value():
    # one hit at rank 2, one relevant item: DCG = 1/log2(3), IDCG = 1
    assert math.isclose(ndcg_at_k(["x", "a"], {"a"}, 2), 1 / math.log2(3))


def test_precision_and_recall():
    ranked = ["a", "x", "b", "y"]
    assert precision_at_k(ranked, {"a", "b", "c"}, 4) == 0.5
    assert math.isclose(recall_at_k(ranked, {"a", "b", "c"}, 4), 2 / 3)
    assert recall_at_k(ranked, set(), 4) == 0.0


def test_top_k_matches_full_sort():
    rng = np.random.default_rng(0)
    q, items = rng.normal(size=(7, 16)), rng.normal(size=(300, 16))
    expected = np.argsort(-(q @ items.T), axis=1)[:, :20]
    np.testing.assert_array_equal(top_k(q, items, 20, batch_size=3), expected)
