"""JSON I/O helpers and user-history loading."""

import json
from pathlib import Path

from .config import RAW_DIR


def load_json(path: Path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def user_history_path(domain: str = "music") -> Path:
    return (
        RAW_DIR / "user_book_history.json"
        if domain == "music"
        else (RAW_DIR / domain / "user_book_history.json")
    )


def load_user_history(domain: str = "music"):
    return load_json(user_history_path(domain))
