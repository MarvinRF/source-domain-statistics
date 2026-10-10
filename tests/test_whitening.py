import numpy as np

from acw.whitening import ZCAWhitening


def _anisotropic(n=2000, d=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d)) @ np.diag(np.linspace(0.1, 5, d)) + 3.0


def test_zca_matrix_is_symmetric_and_whitens():
    x = _anisotropic()
    zca = ZCAWhitening(eps=0).fit(x)
    np.testing.assert_allclose(zca.W_, zca.W_.T, atol=1e-10)
    xc = x - zca.mean_
    cov_w = np.cov((xc @ zca.W_.T).T)
    np.testing.assert_allclose(cov_w, np.eye(x.shape[1]), atol=1e-6)


def test_transform_outputs_unit_norm_rows():
    x = _anisotropic()
    out = ZCAWhitening().fit_transform(x)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
