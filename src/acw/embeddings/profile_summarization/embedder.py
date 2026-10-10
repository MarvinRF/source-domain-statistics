"""Encoding of user profile summaries into embedding matrices."""

import json
import logging
import time

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import PROCESSED_DIR
from .metadata import tagged_stem

log = logging.getLogger(__name__)


def embed_profiles(
    summaries: dict,
    test_users: list[str],
    heuristic: str,
    model: SentenceTransformer,
    dry_run: bool,
    run_tag: str | None = None,
) -> np.ndarray:
    ordered_users = [u for u in test_users if summaries.get(u) is not None]
    texts = [summaries[u] for u in ordered_users]

    log.info(
        f"\nEmbedding {len(texts):,} {heuristic.upper()} profiles "
        f"({len(test_users) - len(ordered_users)} skipped as None)..."
    )

    t0 = time.time()
    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    log.info(f"  Done in {time.time() - t0:.1f}s | shape={embeddings.shape}")

    if not dry_run:
        stem = tagged_stem(heuristic, run_tag)
        index_stem = "user_profile_index_" + stem[len("user_profiles_") :]
        npy_path = PROCESSED_DIR / f"{stem}.npy"
        index_path = PROCESSED_DIR / f"{index_stem}.json"

        np.save(npy_path, embeddings.astype(np.float32))

        index = {uid: idx for idx, uid in enumerate(ordered_users)}
        index_path.write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")

        log.info(f"  Saved: {npy_path.name}")
        log.info(f"  Saved: {index_path.name}")

    return embeddings
