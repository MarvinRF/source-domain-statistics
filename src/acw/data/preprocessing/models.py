# src/acw/data/preprocessing/models.py
"""
Data structures shared across the preprocessing pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Run configuration
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DomainPaths:
    """Resolved paths for a domain pair (source=Books, target=X)."""

    source_reviews: Path  # Books reviews .jsonl.gz
    source_meta: Path  # Books metadata .jsonl.gz
    target_reviews: Path  # Target domain reviews .jsonl.gz
    target_meta: Path  # Target domain metadata .jsonl.gz
    out_dir: Path  # Output directory (data/raw/ for music, data/raw/<domain>/ otherwise)
    target_label: str  # Human-readable label ("album", "movie", "game")
    target_domain: str  # Identifier ("music", "movies", "games")


@dataclass(frozen=True)
class PreprocessingConfig:
    """Pipeline run parameters."""

    min_rating: float = 4.0
    min_interactions: int = 5
    min_history: int = 3
    min_text_len: int = 20
    log_interval: int = 500_000
    max_users: int | None = None  # None = no limit (e.g. 50_000 for debugging)


# ─────────────────────────────────────────────────────────────────────────────
# Intermediate structures
# ─────────────────────────────────────────────────────────────────────────────

# type aliases
UserToItems = dict[str, list[tuple[str, int]]]  # user_id → [(asin, timestamp), ...]
ItemToUsers = dict[str, set[str]]  # asin    → {user_id, ...}


@dataclass
class InteractionData:
    """Output of collect_interactions for one domain."""

    user_to_items: UserToItems
    item_to_users: ItemToUsers
    total_reviews: int
    kept_reviews: int


@dataclass
class KCoreResult:
    """Output of apply_kcore."""

    valid_users: set[str]
    valid_items: set[str]
    iterations: int


@dataclass
class ItemRecord:
    """Metadata of an item after filtering."""

    parent_asin: str
    title: str
    text: str
    average_rating: float | None
    rating_number: int | None
    categories: list[str]


# ─────────────────────────────────────────────────────────────────────────────
# Final result
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class PreprocessingResult:
    """Complete pipeline output for a domain pair."""

    user_book_history: dict[str, list[str]]  # user_id → [book_asin, ...]
    ground_truth: dict[str, list[str]]  # user_id → [target_asin, ...]
    books_meta: list[ItemRecord]
    target_meta: list[ItemRecord]
    stats: dict[str, object]


@dataclass
class HistogramStats:
    min: int
    max: int
    mean: float
    total: int
