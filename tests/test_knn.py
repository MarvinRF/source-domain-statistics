import numpy as np
import scipy.sparse as sp

from experiments.granularity.knn import knn_scores, neighbours, normalise


def test_knn_scores_equal_the_brute_force_similarity_weighted_sum():
    rng = np.random.default_rng(0)
    sim = rng.uniform(-0.2, 1, size=(4, 10)).astype(np.float32)
    inter = sp.csr_matrix((rng.uniform(size=(10, 7)) < 0.4).astype(np.float32))
    idx, w = neighbours(sim, 5)
    assert (np.diff(w, axis=1) <= 0).all()  # most similar first
    for k in (1, 3, 5):
        got = knn_scores(idx, w, k, inter)
        for q in range(4):
            top = np.argsort(-sim[q])[:k]
            want = (sim[q, top, None] * inter[top].toarray()).sum(axis=0)
            np.testing.assert_allclose(got[q], want, atol=1e-6)


def test_normalise_divides_by_the_row_maximum_and_keeps_empty_rows_at_zero():
    out = normalise(np.array([[1.0, 4.0], [0.0, 0.0]], dtype=np.float32))
    assert out.tolist() == [[0.25, 1.0], [0.0, 0.0]]


def test_lambda_tops_plain_at_zero_matches_top_k_scores_and_ties_follow_the_rank():
    import math

    from acw.evaluation.ranking import top_k_scores
    from experiments.granularity.common import lambda_tops

    rng = np.random.default_rng(2)
    s = rng.normal(size=(3, 20)).astype(np.float32)
    aux = np.zeros((3, 20), dtype=np.float32)
    aux[:, 5:9] = 1.0  # four items tie on aux
    rank = rng.permutation(20)
    out = lambda_tops(s, aux, rank, 6, lambdas=(0, math.inf), plain_at_zero=True)
    assert (out[0] == top_k_scores(s, 6)).all()
    for r in range(3):  # lambda = inf: aux first, then the content score (no exact ties)
        want = sorted(range(20), key=lambda i: (-aux[r, i], -s[r, i]))[:6]
        assert out[math.inf][r].tolist() == want
