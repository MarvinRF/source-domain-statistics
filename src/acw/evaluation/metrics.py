"""Ranking metrics (nDCG@k, precision@k, recall@k) with binary relevance.
Dependency-free apart from NumPy."""

from __future__ import annotations

import numpy as np


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    hits = [1 if item in relevant else 0 for item in ranked[:k]]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for item in ranked[:k] if item in relevant) / k


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    return sum(1 for item in ranked[:k] if item in relevant) / len(relevant) if relevant else 0.0
