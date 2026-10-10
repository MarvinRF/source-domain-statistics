# src/acw/data/preprocessing/config.py
"""Default paths, filtering thresholds, logging setup and the registry of
supported target domains for the preprocessing pipeline."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
from acw.settings import ROOT  # noqa: E402

DATA_SETS_DIR = Path.home() / "Documentos" / "data-sets"
RAW_DIR = ROOT / "data" / "raw"

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)

log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────────────────────────────────────

MIN_RATING = 4.0  # minimum rating for a positive interaction
MIN_INTERACTIONS = 5  # k of the iterative k-core
MIN_HISTORY = 3  # minimum books/items per user to be eligible
MIN_TEXT_LEN = 20  # minimum metadata text length (chars) to keep an item
LOG_INTERVAL = 500_000

# ─────────────────────────────────────────────────────────────────────────────
# Known target domains — (subdir, reviews_file, meta_file, label_singular)
# Used by the CLI to resolve paths from --target-domain
# ─────────────────────────────────────────────────────────────────────────────

KNOWN_DOMAINS: dict[str, tuple[str, str, str, str]] = {
    "music": (
        "musicas",
        "CDs_and_Vinyl.jsonl.gz",
        "meta_CDs_and_Vinyl.jsonl.gz",
        "album",
    ),
    "movies": (
        "filmes",
        "Movies_and_TV.jsonl.gz",
        "meta_Movies_and_TV.jsonl.gz",
        "movie",
    ),
    "games": ("videos", "Video_Games.jsonl.gz", "meta_Video_Games.jsonl.gz", "game"),
}
