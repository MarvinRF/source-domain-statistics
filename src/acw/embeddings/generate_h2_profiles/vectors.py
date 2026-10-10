"""Embedding dimension validation and L2 normalization."""

import faiss
import numpy as np


def validate_embedding_dimensions(
    embeddings: np.ndarray,
    expected_dim: int,
):
    real_dim = embeddings.shape[1]

    if real_dim != expected_dim:
        raise RuntimeError(
            "\n"
            "Embedding dimension mismatch detected.\n\n"
            f"Expected dimension : {expected_dim}\n"
            f"Found dimension    : {real_dim}\n\n"
            "Possible causes:\n"
            "  • embeddings generated with another model\n"
            "  • stale embedding artifacts\n"
            "  • inconsistent experiment pipeline\n"
        )


def normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = vectors.astype(np.float32)

    faiss.normalize_L2(vectors)

    return vectors
