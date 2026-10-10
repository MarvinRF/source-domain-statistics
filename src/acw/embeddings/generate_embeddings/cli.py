"""Command-line entry point for item embedding generation.

Encodes item metadata (title + categories, optionally description, or title only) from the
Amazon Reviews 2023 dataset into L2-normalized dense embeddings, producing
FAISS-ready .npy arrays and an item-ID-to-index mapping.

Default-mode outputs (all in data/processed/):
  book_embeddings.npy        — float32 array (N_books,  D)
  album_embeddings.npy       — float32 array (N_albums, D)
  book_id_to_idx.json        — {"ASIN": int_index, ...}
  album_id_to_idx.json       — {"ASIN": int_index, ...}
  books_texts.json           — list of prepared texts (same order as embeddings)
  albums_texts.json          — list of prepared texts (same order as embeddings)
  embedding_stats.json       — run metadata (model, timings, shapes, etc.)

Usage:
  # Default domains (books + albums)
  python -m acw.embeddings.generate_embeddings.cli

  # Different backbone
  python -m acw.embeddings.generate_embeddings.cli \
      --model Alibaba-NLP/gte-Qwen2-7B-instruct --batch-size 32

  # Arbitrary domain (movies, games, etc.)
  python -m acw.embeddings.generate_embeddings.cli \
      --jsonl-path data/raw/movies/movies_filtered.jsonl \
      --out-prefix movies \
      --model Alibaba-NLP/gte-Qwen2-7B-instruct \
      --batch-size 32

  # Title-only text (no categories), e.g. the music albums
  python -m acw.embeddings.generate_embeddings.cli \
      --jsonl-path data/raw/albums_filtered.jsonl \
      --out-prefix albums --out-dir data/processed/title \
      --text-format title

  # Books only / albums only
  python -m acw.embeddings.generate_embeddings.cli --books-only
  python -m acw.embeddings.generate_embeddings.cli --no-gpu --albums-only
"""

import argparse

from .pipeline import run
from .text_prep import TEXT_FNS


def parse_args():
    p = argparse.ArgumentParser(
        description="Generate item embeddings for zero-shot cross-domain recommendation."
    )
    p.add_argument(
        "--model",
        default="all-MiniLM-L12-v2",
        help=(
            "SentenceTransformer model name. "
            "Examples: all-MiniLM-L12-v2 (default), "
            "BAAI/bge-large-en-v1.5, "
            "Alibaba-NLP/gte-Qwen2-7B-instruct"
        ),
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for encoding. Use 32-64 for 7B models.",
    )
    p.add_argument(
        "--no-gpu",
        action="store_true",
        help="Force CPU execution (slower, no GPU memory required).",
    )
    p.add_argument(
        "--books-only",
        action="store_true",
        help="Embed books only (useful for resuming after a crash).",
    )
    p.add_argument(
        "--albums-only",
        action="store_true",
        help="Embed albums only.",
    )
    # ── Arbitrary domain ────────────────────────────────────────────────────────
    p.add_argument(
        "--jsonl-path",
        type=str,
        default=None,
        help=(
            "Path to a .jsonl file with item metadata for any domain. "
            "Requires --out-prefix. When set, --books-only/--albums-only are ignored."
        ),
    )
    p.add_argument(
        "--out-prefix",
        type=str,
        default=None,
        help=(
            "Output file prefix when --jsonl-path is given. "
            "E.g. 'movies' -> data/processed/movies_embeddings_qwen.npy, "
            "movies_id_to_idx_qwen.json, movies_texts.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help=(
            "Output directory for --jsonl-path mode. "
            "Default: data/processed/ (same as the default mode)."
        ),
    )
    p.add_argument(
        "--text-format",
        choices=tuple(TEXT_FNS),
        default="simple",
        help=(
            "Text format for --jsonl-path mode: simple = title+categories; "
            "desc = title+categories+description; title = title only "
            "(an item without a title is embedded as the empty string)."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()
    # --jsonl-path and --out-prefix must be given together.
    if (args.jsonl_path is None) != (args.out_prefix is None):
        raise SystemExit("Error: --jsonl-path and --out-prefix must be used together.")
    run(args)


if __name__ == "__main__":
    main()
