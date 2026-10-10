"""Paths, settings and random seeds for H2 profile generation."""

import random

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
from acw.settings import (
    EMBEDDING_DIM,
    EMBEDDING_MODEL,  # noqa: F401
    MAX_BOOKS_PROFILE,  # noqa: F401
    N_TEST_USERS,  # noqa: F401
    ROOT,  # noqa: E402
    SEED,
)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG — read from acw.settings (overridable via environment variables)
# ─────────────────────────────────────────────────────────────────────────────

EXPECTED_DIM = EMBEDDING_DIM
BOOK_EMBEDDINGS_FILE = "book_embeddings.npy"

# ─────────────────────────────────────────────────────────────────────────────
# RANDOM SEEDS
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
