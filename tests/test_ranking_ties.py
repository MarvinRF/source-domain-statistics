import numpy as np

from acw.evaluation.ranking import top_k_scores_ties


def test_ties_are_broken_by_rank_and_match_a_full_lexsort():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 4, size=(30, 60)).astype(np.float32)  # many exact ties
    tie_rank = rng.permutation(60)
    got = top_k_scores_ties(scores, 10, tie_rank)
    for r in range(30):
        want = np.lexsort((tie_rank, -scores[r]))[:10]
        assert got[r].tolist() == want.tolist()


def test_all_zero_row_returns_the_k_smallest_ranks():
    tie_rank = np.array([3, 0, 4, 1, 2])
    got = top_k_scores_ties(np.zeros((1, 5)), 3, tie_rank)
    assert got[0].tolist() == [1, 3, 4]
