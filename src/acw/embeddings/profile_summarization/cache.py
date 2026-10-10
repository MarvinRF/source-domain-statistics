"""On-disk JSON cache of LLM summaries with periodic atomic flushes."""

import json
import logging
from pathlib import Path

from .config import CHECKPOINT_EVERY

log = logging.getLogger(__name__)


class SummaryCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: dict = {}
        self._pending = 0

        if path.exists():
            try:
                self._data = json.loads(path.read_text(encoding="utf-8"))
                log.info(f"  Cache loaded: {len(self._data):,} entries from {path.name}")
            except json.JSONDecodeError:
                log.warning(f"  Corrupted cache ignored: {path.name}")

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def set(self, key: str, value: str | None):
        self._data[key] = value
        self._pending += 1
        if self._pending >= CHECKPOINT_EVERY:
            self.flush()
            self._pending = 0

    def flush(self):
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self.path)

    def items(self):
        return self._data.items()
