# src/acw/settings.py
"""
Central configuration: paths, model choices and protocol constants. Every
module reads these values instead of hard-coding them; models and endpoints
can be overridden through environment variables without code changes:

    # Local defaults
    python -m acw.embeddings.profile_summarization.cli --heuristic h3

    # Larger models on a GPU server
    export ACW_EMBEDDING_MODEL="Qwen/Qwen3-Embedding-8B"
    export ACW_EMBEDDING_DIM=4096
    export ACW_LLM_MODEL="<name or path of the local model>"
    python -m acw.embeddings.profile_summarization.cli --heuristic h3

Recognised environment variables:
    ACW_DATASETS_DIR     — directory with the Amazon Reviews 2023 downloads
    ACW_EMBEDDING_MODEL  — embedding backbone
    ACW_EMBEDDING_DIM    — backbone output dimension
    ACW_LLM_MODEL        — local LLM (Ollama)
    LLM_BACKEND          — "ollama" | "vllm"
    VLLM_BASE_URL        — vLLM server URL
    VLLM_MODEL_PATH      — model path on the vLLM server
    OLLAMA_URL           — Ollama generate endpoint
    ACW_SPLIT_SEED       — seed of the user split of experiments/granularity (default 42)
"""

from __future__ import annotations

import os
from pathlib import Path

# Repository root and data locations. Every module resolves paths from here.
ROOT: Path = Path(__file__).resolve().parents[2]
RAW_DIR: Path = ROOT / "data" / "raw"
PROCESSED_DIR: Path = ROOT / "data" / "processed"
RESULTS_DIR: Path = ROOT / "results"
# Amazon Reviews 2023 downloads (see docs/DATA.md).
DATASETS_DIR: Path = Path(os.getenv("ACW_DATASETS_DIR", ROOT / "data" / "amazon2023")).expanduser()

# ─────────────────────────────────────────────────────────────────────────────
# Embedding backbone
# ─────────────────────────────────────────────────────────────────────────────

EMBEDDING_MODEL: str = os.getenv(
    "ACW_EMBEDDING_MODEL",
    "Alibaba-NLP/gte-large-en-v1.5",  # canonical backbone of the paper
)

EMBEDDING_DIM: int = int(
    os.getenv(
        "ACW_EMBEDDING_DIM",
        "1024",  # gte-large / bge-large output dimension
    )
)

# ─────────────────────────────────────────────────────────────────────────────
# LLMs
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_LLM_MODEL: str = os.getenv(
    "ACW_LLM_MODEL",
    "qwen2.5:14b",  # local default via Ollama
)

# Backend and endpoints used by the LLM clients
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "ollama")
VLLM_BASE_URL: str = os.getenv("VLLM_BASE_URL", "http://localhost:8000/v1")
VLLM_MODEL_PATH: str = os.getenv(
    "VLLM_MODEL_PATH",
    "llama-3.3-70b-awq",  # name served by vLLM; override with the actual served model
)
OLLAMA_URL: str = os.getenv(
    "OLLAMA_URL",
    "http://localhost:11434/api/generate",
)

# ─────────────────────────────────────────────────────────────────────────────
# Canonical protocol (fixed)
# ─────────────────────────────────────────────────────────────────────────────

SEED: int = 42
# Seed of the user split the granularity experiments read (ACW_SPLIT_SEED); 42 is the main split in
# data/splits/ and results/granularity/, any other seed S lives in the subdirectories seed{S}/.
SPLIT_SEED: int = int(os.getenv("ACW_SPLIT_SEED", SEED))
SPLIT_TAG: str = "" if SPLIT_SEED == SEED else f"seed{SPLIT_SEED}"
GRANULARITY_DIR: Path = RESULTS_DIR / "granularity" / SPLIT_TAG
N_TEST_USERS: int = 2000
MAX_BOOKS_PROFILE: int = 10
