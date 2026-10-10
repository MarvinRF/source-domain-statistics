"""
Integrity and re-encoding check of an item-embedding file (docs/EMBEDDINGS.md).

Reports the shape, dtype and sha256 of the .npy file, the number of non-finite and zero rows and
whether the id map covers rows 0..n-1. Then re-encodes a seeded sample of rows with GTEEncoder,
the text rebuilt from the raw item jsonl with the given --text-format (the formats of the
embedding CLI), and prints the cosine between each fresh vector and the stored row. The file is
"verified" when every sampled cosine is at least --min-cosine (exit status 1 otherwise).

Usage:
    python -m experiments.tools.verify_item_embeddings \\
        --emb title/albums_embeddings_qwen.npy --idx title/albums_id_to_idx_qwen.json \\
        --jsonl data/raw/albums_filtered.jsonl --text-format title

``--emb`` and ``--idx`` are relative to data/processed/. Defaults: 64 rows, seed 42, batch 16.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from acw.embeddings.generate_embeddings.text_prep import TEXT_FNS
from acw.embeddings.profile_summarization.gte_encoder import GTEEncoder
from acw.settings import PROCESSED_DIR


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def item_texts(jsonl: Path, asins: set[str], text_fn) -> dict[str, str]:
    """Text of the items ``asins`` of a filtered item jsonl."""
    out: dict[str, str] = {}
    with open(jsonl, encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            asin = item.get("parent_asin") or item.get("asin", "")
            if asin in asins:
                out[asin] = text_fn(item)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--emb", required=True)
    ap.add_argument("--idx", required=True)
    ap.add_argument("--jsonl", required=True, type=Path)
    ap.add_argument("--text-format", choices=tuple(TEXT_FNS), required=True)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-cosine", type=float, default=0.99999)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    x = np.load(PROCESSED_DIR / args.emb)
    id_map = json.loads((PROCESSED_DIR / args.idx).read_text())
    norms = np.linalg.norm(x, axis=1)
    covered = x.shape[0] == len(id_map) and set(id_map.values()) == set(range(x.shape[0]))
    print(f"file     {args.emb}")
    print(f"sha256   {sha256(PROCESSED_DIR / args.emb)}")
    print(f"shape    {x.shape[0]} x {x.shape[1]} {x.dtype}")
    print(
        f"rows     non-finite {int((~np.isfinite(x).all(axis=1)).sum())}, zero {int((norms == 0).sum())}"
    )
    print(f"id map   covers 0..n-1: {covered}")

    sample = random.Random(args.seed).sample(sorted(id_map), min(args.n, len(id_map)))
    texts = item_texts(args.jsonl, set(sample), TEXT_FNS[args.text_format])
    fresh = GTEEncoder(device=args.device).encode(
        [texts[a] for a in sample], batch_size=args.batch_size
    )
    stored = x[[id_map[a] for a in sample]]
    cos = np.einsum("ij,ij->i", fresh, stored) / (
        np.linalg.norm(fresh, axis=1) * np.linalg.norm(stored, axis=1)
    )
    print(f"sample   {len(sample)} rows, seed {args.seed}, batch {args.batch_size}, {args.device}")
    print(f"cosine   min {cos.min():.7f}  median {np.median(cos):.7f}  max {cos.max():.7f}")
    ok = bool(covered and cos.min() >= args.min_cosine)
    print(f"verified {ok} (every sampled cosine >= {args.min_cosine})")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
