"""Exact top-k retrieval by inner product (cosine on L2-normalized vectors)."""

from __future__ import annotations

import numpy as np


def top_k_scores(scores: np.ndarray, k: int) -> np.ndarray:
    """Indices of the ``k`` highest entries of each row of ``scores``, best first. Shape (n, k)."""
    part = np.argpartition(-scores, k, axis=1)[:, :k]
    order = np.argsort(-np.take_along_axis(scores, part, axis=1), axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def top_k(queries: np.ndarray, items: np.ndarray, k: int, batch_size: int = 512) -> np.ndarray:
    """Indices of the ``k`` highest-scoring items per query, best first. Shape (n_queries, k)."""
    out = [
        top_k_scores(queries[i : i + batch_size] @ items.T, k)
        for i in range(0, len(queries), batch_size)
    ]
    return np.vstack(out) if out else np.empty((0, k), dtype=np.int64)


def top_k_scores_ties(scores: np.ndarray, k: int, tie_rank: np.ndarray) -> np.ndarray:
    """Like :func:`top_k_scores` but equal scores are ordered by ascending ``tie_rank``
    (a permutation of the column indices, e.g. the rank of the item id), so the list
    is fully determined. Rows are handled one at a time after one partial sort."""
    n, m = scores.shape
    kth = np.partition(scores, m - k, axis=1)[:, m - k]  # k-th largest value of each row
    out = np.empty((n, k), dtype=np.int64)
    for r in range(n):
        s, t = scores[r], kth[r]
        hi = np.flatnonzero(s > t)
        eq = np.flatnonzero(s == t)
        need = k - hi.size
        if eq.size > need:
            eq = eq[np.argpartition(tie_rank[eq], need - 1)[:need]]
        cand = np.concatenate([hi, eq])
        out[r] = cand[np.lexsort((tie_rank[cand], -s[cand]))]
    return out
