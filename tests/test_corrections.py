import numpy as np

from acw.corrections import abtt, center, fit_moments, pca_whiten, zca_power
from acw.whitening import ZCAWhitening


def _data(n=1500, d=12, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(n, d)) @ np.diag(np.linspace(0.1, 5, d)) + 3.0).astype(np.float32)


def _unit(x):
    np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1.0, atol=1e-5)
    assert x.dtype == np.float32


def test_zca_alpha_one_matches_zca_whitening():
    x = _data()
    m = fit_moments(x)
    out = zca_power(x, *m, 1.0)
    _unit(out)
    np.testing.assert_allclose(out, ZCAWhitening().fit_transform(x), atol=1e-4)


def test_zca_alpha_zero_is_centering():
    x = _data()
    m = fit_moments(x)
    out = zca_power(x, *m, 0.0)
    np.testing.assert_allclose(out, center(x, m.mean), atol=1e-5)


def test_pca_is_a_rotation_of_zca():
    x = _data()
    m = fit_moments(x)
    # Same whitening up to the rotation V^T, so pairwise cosines are identical.
    np.testing.assert_allclose(pca_whiten(x, *m) @ m.eigvecs.T, zca_power(x, *m, 1.0), atol=1e-4)


def test_abtt_removes_top_directions():
    x = _data()
    m = fit_moments(x)
    out = abtt(x, m.mean, m.eigvecs, 3)
    _unit(out)
    assert np.abs(out @ m.eigvecs[:, :3]).max() < 1e-5
    assert np.abs(out @ m.eigvecs[:, 3:]).max() > 1e-2


def test_fit_moments_chunking_is_exact():
    x = _data()
    a, b = fit_moments(x), fit_moments(x, chunk=100)
    np.testing.assert_allclose(a.mean, b.mean)
    np.testing.assert_allclose(a.eigvals, b.eigvals, rtol=1e-9)
    assert np.all(np.diff(a.eigvals) <= 0)
