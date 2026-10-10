import math

import numpy as np

from experiments.granularity.hybrid import hybrid_top, popularity_z


def test_lambda_zero_is_content_and_inf_is_popularity_with_cosine_ties():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(40, 8)).astype(np.float32)
    x /= np.linalg.norm(x, axis=1, keepdims=True)
    q = x[:5] + 0.1
    counts = rng.integers(0, 6, size=40)  # many exact ties
    pz = popularity_z(counts)
    out = hybrid_top(q, x, pz, lambdas=(0, 1, math.inf), k=10)
    cos = q @ x.T
    np.testing.assert_array_equal(out[0], np.argsort(-cos, axis=1, kind="stable")[:, :10])
    for u in range(5):  # popularity first, then cosine inside equal counts
        order = sorted(range(40), key=lambda i: (-counts[i], -cos[u, i]))
        assert list(out[math.inf][u]) == order[:10]


def test_popularity_z_is_standardised():
    pz = popularity_z(np.array([0, 1, 3, 10, 100]))
    assert abs(pz.mean()) < 1e-6 and abs(pz.std() - 1) < 1e-6
