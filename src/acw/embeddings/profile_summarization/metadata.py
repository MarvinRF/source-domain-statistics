"""Output filename tagging, per-run metadata and the append-only run manifest."""

import json
import logging
from datetime import UTC, datetime

from .config import (
    MAX_BOOKS_PROFILE,
    PROCESSED_DIR,
    TEMPERATURE_H3,
    TEMPERATURE_H3_CONCRETE,
    TEMPERATURE_H4,
    TEMPERATURE_H4_CONCRETE,
)

log = logging.getLogger(__name__)

RUN_MANIFEST_PATH = PROCESSED_DIR / "run_manifest.jsonl"


def tagged_stem(heuristic: str, run_tag: str | None) -> str:
    """Filename stem for this heuristic's outputs.

    Untagged runs use the default ``user_profiles_{heuristic}`` name, which
    the canonical evaluation pipeline reads. Tagged runs get a
    distinct ``__{run_tag}`` suffix so a different model/run can never
    silently merge into or overwrite a previous run's cache.
    """
    return f"user_profiles_{heuristic}" + (f"__{run_tag}" if run_tag else "")


def append_run_manifest(record: dict):
    """Append-only log of every generation run, one JSON object per line.

    Never truncates or overwrites — safe to call from concurrent/sequential
    runs. This is the source of truth for "which model produced which file",
    independent of whether the run used --run-tag.
    """
    RUN_MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RUN_MANIFEST_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"  Run manifest appended: {RUN_MANIFEST_PATH.name}")


def save_metadata(
    heuristic: str,
    summaries: dict,
    embedding_shape: tuple | None,
    llm_model: str,
    embedding_model: str | None,
    run_tag: str | None = None,
    text_only: bool = False,
):
    valid = sum(1 for v in summaries.values() if v is not None)
    skipped = len(summaries) - valid
    generated_at = datetime.now(UTC).isoformat()

    metadata = {
        "heuristic": heuristic,
        "run_tag": run_tag,
        "generated_at": generated_at,
        "llm_model": llm_model,
        "embedding_model": None if text_only else embedding_model,
        "embedding_dimension": None
        if (text_only or embedding_shape is None)
        else embedding_shape[1],
        "text_only": text_only,
        "users_total": len(summaries),
        "users_valid": valid,
        "users_skipped": skipped,
        "max_books_profile": MAX_BOOKS_PROFILE,
        "temperatures": {
            "h3": TEMPERATURE_H3,
            "h4": TEMPERATURE_H4,
            "h3_concrete": TEMPERATURE_H3_CONCRETE,
            "h4_concrete": TEMPERATURE_H4_CONCRETE,
        },
    }

    stem = tagged_stem(heuristic, run_tag)
    path = PROCESSED_DIR / f"{stem}_metadata.json"
    path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info(f"  Metadata saved: {path.name}")

    append_run_manifest({**metadata, "metadata_file": path.name})
