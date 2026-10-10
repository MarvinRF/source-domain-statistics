"""JSONL loader that turns item metadata records into (ID, text) pairs."""

import json
import logging
from pathlib import Path

from tqdm import tqdm

log = logging.getLogger(__name__)


def load_jsonl(
    path: Path,
    text_fn,
    domain_label: str,
    keep_empty: bool = False,
) -> tuple[list[str], list[str], list[str]]:
    """
    Reads a .jsonl file and returns three parallel lists:
      asin_list   — item IDs (parent_asin or asin)
      text_list   — prepared text strings
      empty_list  — ASINs where text preparation returned empty string

    An item whose text is empty falls back to ``"<label>: <title>"`` and is skipped when it has
    no title. With ``keep_empty`` it is kept with the empty text instead (for formats that are
    empty exactly when the title is, so that rows stay aligned with the catalog).
    """
    asin_list, text_list, empty_list = [], [], []
    skipped = 0

    log.info(f"Loading {domain_label} from {path.name} ...")
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"  Reading {domain_label}", unit="items"):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue

            asin = item.get("parent_asin") or item.get("asin", "")
            if not asin:
                skipped += 1
                continue

            text = text_fn(item)
            if not text:
                empty_list.append(asin)
                fallback = (item.get("title") or "").strip()
                if fallback:
                    text = f"{domain_label}: {fallback}"
                elif not keep_empty:
                    skipped += 1
                    continue

            asin_list.append(asin)
            text_list.append(text)

    log.info(
        f"  {domain_label}: {len(asin_list):,} items loaded | "
        f"{len(empty_list):,} with sparse metadata | "
        f"{skipped:,} skipped (no ASIN/text)"
    )
    return asin_list, text_list, empty_list
