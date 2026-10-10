"""
experiments/robustness/offdomain_catalog_build.py
=================================================
Builds an item catalog from an Amazon Reviews 2023 category that has NO
genre/subgenre/mood taxonomy (Grocery_and_Gourmet_Food), for the structural
confound test of the "encoder-level, not catalog-specific" claim. That claim
was originally tested only between two "creative" catalogs (books, albums),
both organised by genre/subgenre/mood, so it could not rule out that the
shared geometry reflects the structure of cultural catalogs rather than a
property of the encoder itself.

Reuses `_build_item_text` from src/acw/data/preprocessing/pipeline.py (the
same function used for books/movies/games); the text-construction logic is
not duplicated.

Source: raw/meta_categories/meta_Grocery_and_Gourmet_Food.jsonl from the
McAuley-Lab/Amazon-Reviews-2023 dataset (HF Hub). A ~400MB prefix (the first
~170k lines of the ~1.4GB file) is expected at
data/raw/grocery/meta_grocery_raw_sample.jsonl (e.g. downloaded with an HTTP
Range request). It is a prefix, not the full category, which is sufficient:
the dump is not ordered by popularity or quality, so the prefix introduces no
text- or quality-related selection bias. The catalog is then subsampled to
the size of the album catalog for a size-matched comparison.

Supports: the off-domain (non-cultural catalog) control of the encoder-level
geometry claim; its output is consumed by
experiments/robustness/offdomain_covariance.py.

Input:  data/raw/grocery/meta_grocery_raw_sample.jsonl
Output: data/raw/grocery/grocery_filtered.jsonl

Usage:
    python -m experiments.robustness.offdomain_catalog_build
"""

import json
import random

from acw.data.preprocessing.pipeline import _build_item_text
from acw.settings import ROOT  # noqa: E402

SRC = ROOT / "data" / "raw" / "grocery" / "meta_grocery_raw_sample.jsonl"
OUT = ROOT / "data" / "raw" / "grocery" / "grocery_filtered.jsonl"

MIN_TEXT_LEN = 20
TARGET_N = 67_986  # album catalog size, for a size-matched comparison
SEED = 42


def main():
    records = []
    n_total = 0
    n_bad_json = 0
    with open(SRC, encoding="utf-8") as f:
        for line in f:
            n_total += 1
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                n_bad_json += 1
                continue
            text = _build_item_text(d)
            if len(text) < MIN_TEXT_LEN:
                continue
            asin = d.get("parent_asin")
            if not asin:
                continue
            records.append(
                {
                    "parent_asin": asin,
                    "title": d.get("title", ""),
                    "text": text,
                    "average_rating": d.get("average_rating"),
                    "rating_number": d.get("rating_number"),
                    "categories": d.get("categories", []),
                }
            )

    print(f"Lines read: {n_total:,} (invalid/truncated JSON at the end: {n_bad_json:,})")
    print(f"Valid items (text >= {MIN_TEXT_LEN} chars): {len(records):,}")

    # dedup by parent_asin (the dump may repeat items across variants)
    seen = set()
    dedup = []
    for r in records:
        if r["parent_asin"] in seen:
            continue
        seen.add(r["parent_asin"])
        dedup.append(r)
    print(f"After dedup by parent_asin: {len(dedup):,}")

    rng = random.Random(SEED)
    if len(dedup) > TARGET_N:
        sample = rng.sample(dedup, TARGET_N)
    else:
        sample = dedup
    print(f"Final sample (seed={SEED}, target={TARGET_N:,}): {len(sample):,}")

    # qualitative check: confirm there is no genre/mood taxonomy
    cats = set()
    for r in sample[:2000]:
        for c in r["categories"]:
            cats.add(c)
    print("\nSample of categories present (manual check for absence of genre/mood):")
    for c in sorted(cats)[:40]:
        print(f"  - {c}")

    with open(OUT, "w", encoding="utf-8") as f:
        for r in sample:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
