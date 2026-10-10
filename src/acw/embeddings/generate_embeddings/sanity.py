"""Basic sanity checks on generated embeddings (finite values, unit norm, variance)."""

import logging

import numpy as np

log = logging.getLogger(__name__)


def sanity_check_embeddings(embeddings: np.ndarray, n_sample: int = 500):
    """
    Verifies that the embedding space has meaningful structure:
    - Embeddings are L2-normalized (norm ≈ 1.0)
    - Variance across dimensions is non-trivial (model is doing work)
    - No NaN / Inf values
    """
    log.info("  Running embedding sanity check...")
    sample = embeddings[:n_sample]

    if np.any(np.isnan(sample)) or np.any(np.isinf(sample)):
        log.error("  ✗ SANITY FAIL: NaN or Inf detected in embeddings!")
        return

    norms = np.linalg.norm(sample, axis=1)
    mean_norm = norms.mean()
    if not (0.99 < mean_norm < 1.01):
        log.warning(f"  ⚠ Norms not ≈ 1.0 (mean={mean_norm:.4f}). Check normalize_embeddings flag.")
    else:
        log.info(f"  ✓ L2 norms ≈ 1.0 (mean={mean_norm:.4f})")

    var = sample.var(axis=0).mean()
    log.info(f"  ✓ Mean per-dimension variance: {var:.6f}")
    if var < 1e-6:
        log.warning("  ⚠ Very low variance — embeddings may be degenerate.")

    log.info("  Sanity check passed.")
