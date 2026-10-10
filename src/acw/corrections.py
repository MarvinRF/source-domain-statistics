"""Training-free geometric corrections of frozen text embeddings (pure numpy).

Every transform is *fitted* on a reference matrix (an item catalog, a book
catalog, or the profiles of the training users) with :func:`fit_moments` and
*applied* to any matrix with a function below. Outputs are float32 with
L2-normalized rows, so cosine similarity is a plain inner product.

========================  =====================================================
``center``                subtract the fitted mean
``abtt``                  All-but-the-Top: center, then remove the projections
                          on the top-D principal directions
``pca_whiten``            Λ^-1/2 Vᵀ (decorrelates, changes the basis)
``zca_power``             V diag((λ+eps)^(-α/2)) Vᵀ, same basis as the input;
                          α=1 is ZCA whitening, α=0 is centering only
========================  =====================================================

Exponent convention: ``alpha`` scales the *whitening strength*, so the
eigenvalue exponent is ``-alpha/2`` (λ^-1/2 at ``alpha=1``, the usual ZCA);
``alpha`` is not the eigenvalue exponent itself.

The ZCA path reproduces :class:`acw.whitening.ZCAWhitening` (sample covariance
with N-1 denominator, float64 eigendecomposition, ``eps=1e-6`` added to the
eigenvalues).
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np

EPS = 1e-6
_CHUNK = 20_000  # rows per float64 block; bounds the transient memory of every pass


class Moments(NamedTuple):
    """First and second moments of a fitted matrix.

    ``mean`` has shape (d,); ``eigvals`` (d,) is sorted descending and
    ``eigvecs`` (d, d) holds the matching eigenvectors as columns.
    """

    mean: np.ndarray
    eigvals: np.ndarray
    eigvecs: np.ndarray


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization, float32 output (zero rows stay zero)."""
    X = np.asarray(X, dtype=np.float32)
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)


def fit_moments(X: np.ndarray, chunk: int = _CHUNK) -> Moments:
    """Mean and eigendecomposition of the sample covariance of ``X`` (N, d).

    The covariance is accumulated in float64 over row blocks, so ``X`` can stay
    float32 and large (e.g. 200k x 4096) without a float64 copy.
    """
    n, d = X.shape
    mean = np.zeros(d, dtype=np.float64)
    for i in range(0, n, chunk):
        mean += X[i : i + chunk].sum(axis=0, dtype=np.float64)
    mean /= n
    cov = np.zeros((d, d), dtype=np.float64)
    for i in range(0, n, chunk):
        block = X[i : i + chunk].astype(np.float64) - mean
        cov += block.T @ block
    cov /= max(n - 1, 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    return Moments(mean, np.clip(eigvals[order], 0.0, None), eigvecs[:, order])


def _apply(X: np.ndarray, mean: np.ndarray, matrix: np.ndarray | None, chunk: int) -> np.ndarray:
    """``l2((X - mean) @ matrix)`` (``matrix=None``: identity) in float64 blocks."""
    out = np.empty(X.shape if matrix is None else (X.shape[0], matrix.shape[1]), dtype=np.float32)
    for i in range(0, X.shape[0], chunk):
        block = X[i : i + chunk].astype(np.float64) - mean
        if matrix is not None:
            block = block @ matrix
        norms = np.maximum(np.linalg.norm(block, axis=1, keepdims=True), 1e-12)
        out[i : i + chunk] = block / norms
    return out


def center(X: np.ndarray, mean: np.ndarray, chunk: int = _CHUNK) -> np.ndarray:
    """Subtract ``mean``, then L2-normalize."""
    return _apply(X, mean, None, chunk)


def abtt(
    X: np.ndarray, mean: np.ndarray, components: np.ndarray, D: int, chunk: int = _CHUNK
) -> np.ndarray:
    """All-but-the-Top: center, remove the projections on the first ``D``
    columns of ``components`` (principal directions, descending), L2-normalize."""
    U = components[:, :D]
    return _apply(X, mean, np.eye(U.shape[0]) - U @ U.T, chunk)


def pca_whiten(
    X: np.ndarray,
    mean: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    eps: float = EPS,
    chunk: int = _CHUNK,
) -> np.ndarray:
    """Full-rank PCA whitening ``(X - mean) V Λ^-1/2``, L2-normalized."""
    return _apply(X, mean, eigvecs / np.sqrt(eigvals + eps), chunk)


def zca_power(
    X: np.ndarray,
    mean: np.ndarray,
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    alpha: float = 1.0,
    eps: float = EPS,
    chunk: int = _CHUNK,
) -> np.ndarray:
    """Partial ZCA whitening ``(X - mean) V diag((λ+eps)^(-α/2)) Vᵀ``, L2-normalized.

    ``alpha=1`` is ZCA (covariance -> identity); ``alpha=0`` only centers.
    """
    W = (eigvecs * (eigvals + eps) ** (-alpha / 2.0)) @ eigvecs.T
    return _apply(X, mean, W, chunk)
