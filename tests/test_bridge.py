import numpy as np

from experiments.granularity.bridge import fit_ridge_path, map_profiles


def test_ridge_path_matches_the_normal_equations_and_recovers_a_linear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(200, 6))
    Y = X @ rng.normal(size=(6, 4)) + 0.3
    mx, my, Ws = fit_ridge_path(X, Y, scales=(1e-8, 1.0))
    Xc, Yc = X - mx, Y - my
    unit = np.trace(Xc.T @ Xc) / 6
    want = np.linalg.solve(Xc.T @ Xc + unit * np.eye(6), Xc.T @ Yc)
    np.testing.assert_allclose(Ws[1.0], want, atol=1e-8)
    np.testing.assert_allclose(map_profiles(X, mx, my, Ws[1e-8]), Y, atol=1e-4)
