"""Copy albums_filtered.jsonl / books_filtered.jsonl, adding the `description`
field from the raw Amazon Reviews 2023 metadata.

Output: data/raw/desc/{albums,books}_desc.jsonl (the canonical files are left
untouched).

Usage:
    python -m acw.data.add_descriptions [--datasets-dir ~/Documentos/data-sets]
"""

import argparse
import gzip
import json
from pathlib import Path

from acw.settings import ROOT  # noqa: E402

RAW = ROOT / "data" / "raw"
PAIRS = {  # filtered jsonl -> raw meta
    "albums": ("albums_filtered.jsonl", "musicas/meta_CDs_and_Vinyl.jsonl.gz"),
    "books": ("books_filtered.jsonl", "livros/meta_Books.jsonl.gz"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets-dir", type=Path, default=Path.home() / "Documentos" / "data-sets")
    args = ap.parse_args()
    out_dir = RAW / "desc"
    out_dir.mkdir(exist_ok=True)

    for name, (filtered, meta) in PAIRS.items():
        items = [json.loads(line) for line in open(RAW / filtered, encoding="utf-8")]
        wanted = {it["parent_asin"] for it in items}
        desc: dict[str, list] = {}
        with gzip.open(args.datasets_dir / meta, "rt", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if rec.get("parent_asin") in wanted:
                    desc[rec["parent_asin"]] = rec.get("description") or []
        out = out_dir / f"{name}_desc.jsonl"
        with open(out, "w", encoding="utf-8") as f:
            for it in items:
                it["description"] = desc.get(it["parent_asin"], [])
                f.write(json.dumps(it, ensure_ascii=False) + "\n")
        n_desc = sum(1 for it in items if any(d.strip() for d in desc.get(it["parent_asin"], [])))
        print(
            f"{name}: {len(items):,} items, {n_desc:,} with non-empty description "
            f"({n_desc / len(items):.1%}) -> {out}"
        )


if __name__ == "__main__":
    main()
