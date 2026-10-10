"""
Category taxonomy audit of the three target catalogs (no embeddings, no users).

The relevance levels ``d1``, ``d2``, ``d3`` of the granularity ladder are positions 1, 2, 3 of
each item's category path, clamped at the deepest position an item has (position 0 is the
store department for most items; see ``position0``). This script describes what those
labels are, per domain, for the items of the evaluated catalog (the items with a GTE
embedding): the most frequent label at each position and its share, the number of distinct
labels, the share of items whose position 0 is not the modal department, and the share of
items whose path stops before the position.

Output: results/granularity/taxonomy.json

Usage:
    python -m experiments.granularity.taxonomy
"""

from __future__ import annotations

import json
from collections import Counter

from acw.evaluation.config import resolve_paths
from acw.provenance import provenance
from acw.settings import PROCESSED_DIR
from experiments.granularity.run_corrections import OUT_DIR, REGISTRY

DOMAINS = ("music", "movies", "games")
POSITIONS = (0, 1, 2, 3)
TOP = 5


def summarise(paths: list[list[str]]) -> dict:
    """Per position: distinct labels, the ``TOP`` most frequent with their share of all
    items, and the share of items whose path has no label at the position."""
    n = len(paths)
    out: dict = {"n_items": n, "positions": {}}
    for k in POSITIONS:
        labels = Counter(p[k] for p in paths if len(p) > k)
        out["positions"][str(k)] = {
            "n_labels": len(labels),
            "share_without_label": 1 - sum(labels.values()) / n,
            "top": [[lab, c / n] for lab, c in labels.most_common(TOP)],
        }
    modal = out["positions"]["0"]["top"][0]
    out["position0"] = {
        "modal_label": modal[0],
        "share_modal": modal[1],
        "n_not_modal": n - round(modal[1] * n),
    }
    return out


def run() -> None:
    result: dict = {"domains": {}}
    inputs = []
    for d in DOMAINS:
        src, cfg = REGISTRY[(d, "gte", "titlecat")], resolve_paths(d)
        catalog = set(json.loads((PROCESSED_DIR / src.items_idx).read_text()))
        inputs += [PROCESSED_DIR / src.items_idx, cfg.items_meta_path]
        paths = []
        with open(cfg.items_meta_path) as f:
            for line in f:
                item = json.loads(line)
                if item.get("parent_asin") in catalog and item.get("categories"):
                    paths.append(item["categories"])
        result["domains"][d] = summarise(paths)
        result["domains"][d]["n_catalog"] = len(catalog)
        print(d, result["domains"][d]["n_items"], result["domains"][d]["position0"])
    result["provenance"] = provenance(inputs)
    (OUT_DIR / "taxonomy.json").write_text(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    run()
