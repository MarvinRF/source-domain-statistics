import numpy as np

from experiments.granularity.controls import derangement


def test_derangement_is_a_seeded_permutation_without_fixed_points():
    for n in (2, 3, 50, 1000):
        p = derangement(n, seed=1)
        assert sorted(p) == list(range(n)) and not (p == np.arange(n)).any()
    assert (derangement(100) == derangement(100)).all()
