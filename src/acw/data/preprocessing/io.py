# src/acw/data/preprocessing/io.py
"""
I/O helpers for preprocessing: streaming reads of .jsonl.gz files and atomic
JSON / JSONL writes.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Generator
from pathlib import Path

log = logging.getLogger(__name__)


def stream_jsonl_gz(
    path: Path,
    max_lines: int | None = None,
) -> Generator[dict, None, None]:
    """
    Stream a .jsonl.gz file line by line without loading it into memory.
    Malformed lines are silently skipped.
    """
    count = 0
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            yield record
            count += 1
            if max_lines is not None and count >= max_lines:
                break


def save_json(obj: object, path: Path) -> None:
    """Write ``obj`` as JSON atomically (via a .tmp file)."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"  Saved: {path.name} ({size_mb:.1f} MB)")


def save_jsonl(records: list[object], path: Path) -> None:
    """Write a list of objects as .jsonl atomically."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    tmp.replace(path)
    size_mb = path.stat().st_size / 1_048_576
    log.info(f"  Saved: {path.name} ({size_mb:.1f} MB)")
