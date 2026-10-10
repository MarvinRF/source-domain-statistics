"""Provenance record stored under the ``"provenance"`` key of every result JSON.

``provenance(input_paths)`` returns the git commit and dirty flag of the code,
the Python and key package versions, a UTC timestamp and the sha256 (plus size)
of each input file, so that a result can be traced to the exact code and data
that produced it. Large files (embeddings) are hashed in streaming chunks.

The dirty flag is true when ``src/``, ``experiments/`` or ``pyproject.toml``
have uncommitted changes (modified or untracked files); data and results do
not count. It is null when git is unavailable.
"""

from __future__ import annotations

import hashlib
import subprocess
import sys
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from acw.settings import ROOT

PACKAGES = ("numpy", "scipy", "pandas", "torch", "transformers", "sentence-transformers")
CODE_PATHS = ("src", "experiments", "pyproject.toml")
CHUNK = 16 * 1024 * 1024  # bytes per read when hashing


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(CHUNK):
            h.update(block)
    return h.hexdigest()


def _git(*args: str) -> str | None:
    try:
        out = subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True, timeout=60
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip()


def _version(pkg: str) -> str | None:
    try:
        return version(pkg)
    except PackageNotFoundError:
        return None


def provenance(input_paths: list[Path | str] | tuple[Path | str, ...] = ()) -> dict:
    """Provenance of a run that read ``input_paths`` (relative paths are taken
    from the repository root; paths inside it are recorded relative to it)."""
    status = _git("status", "--porcelain", "--", *CODE_PATHS)
    inputs = {}
    for p in dict.fromkeys(Path(p) for p in input_paths):
        path = p if p.is_absolute() else ROOT / p
        name = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        inputs[name] = {"sha256": sha256_file(path), "bytes": path.stat().st_size}
    return {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_dirty": None if status is None else bool(status),
        "python": sys.version.split()[0],
        "packages": {pkg: _version(pkg) for pkg in PACKAGES},
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "inputs": inputs,
    }
