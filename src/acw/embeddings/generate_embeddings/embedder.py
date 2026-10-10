"""Batched sentence encoding with periodic on-disk checkpoints for resumable runs."""

import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .config import CHECKPOINT_EVERY

log = logging.getLogger(__name__)


def embed_with_checkpoint(
    texts: list[str],
    model: SentenceTransformer,
    checkpoint_path: Path,
    batch_size: int = 128,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Encodes a list of texts into embeddings, saving a checkpoint every
    CHECKPOINT_EVERY batches. If a checkpoint exists, resumes from it —
    so a crashed run does not lose progress.

    Returns a float32 numpy array of shape (N, D).
    """
    n = len(texts)

    # Try to resume from checkpoint
    start_idx = 0
    if checkpoint_path.exists():
        ckpt = np.load(checkpoint_path)
        start_idx = ckpt.shape[0]
        log.info(f"  Resuming from checkpoint: {start_idx:,} / {n:,} already embedded")
        all_embeddings = list(ckpt)
    else:
        all_embeddings = []

    if start_idx >= n:
        log.info("  All items already embedded (checkpoint complete).")
        return np.array(all_embeddings, dtype=np.float32)

    remaining_texts = texts[start_idx:]
    batches = [
        remaining_texts[i : i + batch_size] for i in range(0, len(remaining_texts), batch_size)
    ]

    batch_iter = tqdm(batches, desc="  Embedding", unit="batch") if show_progress else batches

    for b_idx, batch in enumerate(batch_iter):
        batch_emb = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalize for cosine via dot product
        )
        all_embeddings.extend(batch_emb)

        if (b_idx + 1) % CHECKPOINT_EVERY == 0:
            ckpt_arr = np.array(all_embeddings, dtype=np.float32)
            np.save(checkpoint_path, ckpt_arr)

    final = np.array(all_embeddings, dtype=np.float32)
    np.save(checkpoint_path, final)
    return final
