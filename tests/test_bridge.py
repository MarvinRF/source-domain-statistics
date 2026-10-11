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


def test_contrastive_bridge_learns_and_helpers_are_consistent():
    import scipy.sparse as sp

    from experiments.granularity.bridge import fit_contrastive, holdout, rank_ndcg10, val_ndcg10

    rng = np.random.default_rng(0)
    items = rng.normal(size=(40, 6))
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    own = rng.integers(0, 40, 700)
    X = (items[own] + 0.1 * rng.normal(size=(700, 6))).astype(np.float32)
    inter = sp.csr_matrix((np.ones(700), (np.arange(700), own)), shape=(700, 40))
    f, ep, best = fit_contrastive(X[:600], inter[:600], items, X[600:], inter[600:], 32, 0.1, 0.0, 0)
    top = np.argsort(-(f(X[600:]) @ items.T), axis=1)[:, :10]
    assert ep >= 1 and best > 0.5 and abs(val_ndcg10(top, inter[600:]) - best) < 1e-6  # >> random
    fit, rows, cols = holdout(sp.vstack([inter[:5], inter[:5]]).tocsr(), 0)  # 1 item per user
    assert len(rows) == 0 and fit.nnz == 10
    assert rank_ndcg10(items[[3]], items, np.array([3]))[0] == 1.0
