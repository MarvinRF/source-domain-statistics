import numpy as np

from experiments.granularity.lexical import LexIndex
from experiments.granularity.lexical_overlap import overlap_row, rare_items, reachable, strata_of


def test_overlap_row_chance_levels_and_strata():
    a = np.array([1, 1, 1, 0, 0, 0, 0, 0], dtype=bool)
    b = np.array([1, 1, 0, 1, 0, 0, 0, 0], dtype=bool)
    r = overlap_row(a, b, np.zeros(8, dtype=int))
    assert (r["n"], r["n_a"], r["n_b"], r["both"]) == (8, 3, 3, 2)
    assert r["expected_independent"] == 9 / 8 and r["expected_stratified"] == 9 / 8
    assert r["jaccard"] == 2 / 4 and r["share_a_in_b"] == 2 / 3
    # inside two strata the chance level is the sum over strata
    s = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    assert overlap_row(a, b, s)["expected_stratified"] == 3 * 3 / 4 + 0


def test_strata_follow_the_number_of_relevant_items():
    assert strata_of(np.array([1, 2, 3, 4, 5, 9, 10, 40])).tolist() == [0, 1, 2, 2, 3, 3, 4, 4]


def test_reachable_uses_only_rare_tokens():
    docs = [["the", "beatles"], ["the", "stones"], ["the", "who"], ["the", "kinks"]]
    index = LexIndex(docs)
    rare = rare_items(index, 0.25)  # df <= 1 document: "the" (df 4) is not rare
    q = index.query([["the"], ["the", "beatles"], ["pink"]])
    got = reachable(q, rare)
    assert got.tolist() == [[False] * 4, [True, False, False, False], [False] * 4]
