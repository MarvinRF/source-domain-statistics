"""Paths, logging setup, constants and sampling temperatures for profile summarization."""

import logging
import os
import sys

# ─────────────────────────────────────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────────────────────────────────────
from acw.settings import (
    DEFAULT_LLM_MODEL,  # noqa: F401
    EMBEDDING_MODEL,  # noqa: F401
    LLM_BACKEND,  # noqa: F401
    MAX_BOOKS_PROFILE,  # noqa: F401
    N_TEST_USERS,  # noqa: F401
    OLLAMA_URL,  # noqa: F401
    ROOT,  # noqa: E402
    SEED,  # noqa: F401
    VLLM_BASE_URL,  # noqa: F401
    VLLM_MODEL_PATH,  # noqa: F401
)

RAW_DIR = ROOT / "data" / "raw"
PROCESSED_DIR = ROOT / "data" / "processed"
TRACE_DIR = PROCESSED_DIR / "debug_traces"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TRACE_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

MAX_BOOK_DESC = 240
CHECKPOINT_EVERY = 100
REQUEST_TIMEOUT = 120
MIN_H4_ENTITIES = 3  # minimum number of intent entities required for an H4 profile
MAX_WORKERS = 4

# Maximum number of concurrent LLM requests. Keep it low (4) for a local
# Ollama server; for vLLM with --max-num-seqs N, use about N-4 to leave
# headroom (e.g. ACW_LLM_CONCURRENCY=24 for --max-num-seqs 32).
LLM_CONCURRENCY = int(os.getenv("ACW_LLM_CONCURRENCY", str(MAX_WORKERS)))

# ─────────────────────────────────────────────────────────────────────────────
# TEMPERATURES
# ─────────────────────────────────────────────────────────────────────────────

TEMPERATURE_H3 = 0.35
TEMPERATURE_H4 = 0.10
TEMPERATURE_H3_CONCRETE = 0.05
TEMPERATURE_H4_CONCRETE = 0.05
