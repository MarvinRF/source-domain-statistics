"""Embedding generation pipeline: load item metadata, encode it and save the arrays.

Invoked through ``python -m acw.embeddings.generate_embeddings.cli``.
"""

import json
import logging
import time

import numpy as np

from .config import PROCESSED_DIR, RAW_DIR, log
from .embedder import embed_with_checkpoint
from .loader import load_jsonl
from .sanity import sanity_check_embeddings
from .text_prep import TEXT_FNS, prepare_album_text, prepare_book_text

log = logging.getLogger(__name__)  # noqa: F811


def run(args):
    log.info("=" * 65)
    log.info("Embedding Generation (Real Data)")
    log.info("=" * 65)
    log.info(f"  Model      : {args.model}")
    log.info(f"  Batch size : {args.batch_size}")
    log.info(f"  Device     : {'CPU (forced)' if args.no_gpu else 'auto (GPU if available)'}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    import torch

    device = "cpu" if args.no_gpu else ("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Using device: {device}")

    t0 = time.time()
    if args.model == "Alibaba-NLP/gte-large-en-v1.5":
        # GTE-large's remote code can silently corrupt embeddings through its
        # RoPE position buffer; GTEEncoder works around it (see
        # profile_summarization/gte_encoder.py).
        from ..profile_summarization.gte_encoder import EMBEDDING_DIM, GTEEncoder

        model = GTEEncoder(args.model, device=device)
        dim = EMBEDDING_DIM
    else:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(args.model, device=device, trust_remote_code=True)
        dim = model.get_embedding_dimension()
    log.info(f"  Embedding dimension: {dim}")
    log.info(f"  Model loaded in {time.time() - t0:.1f}s")

    stats = {
        "model": args.model,
        "embedding_dim": dim,
        "device": device,
        "batch_size": args.batch_size,
    }

    # ------------------------------------------------------------------
    # 2a. Arbitrary domain (--jsonl-path + --out-prefix)
    # ------------------------------------------------------------------
    if args.jsonl_path is not None:
        from pathlib import Path as _Path

        jsonl_path = _Path(args.jsonl_path)
        out_dir = _Path(args.out_dir) if args.out_dir else PROCESSED_DIR
        out_dir.mkdir(parents=True, exist_ok=True)
        prefix = args.out_prefix

        log.info("")
        log.info("─" * 65)
        log.info(f"DOMAIN: {prefix.upper()} ({jsonl_path.name})")
        log.info("─" * 65)

        if not jsonl_path.exists():
            raise FileNotFoundError(f"File not found: {jsonl_path}")

        # All text functions are domain-agnostic ("simple" = title + categories). The title-only
        # format is empty when the title is, and such items stay in the catalog order.
        text_format = getattr(args, "text_format", "simple")
        asins, texts, empty = load_jsonl(
            jsonl_path, TEXT_FNS[text_format], prefix, keep_empty=text_format == "title"
        )

        id_to_idx = {asin: idx for idx, asin in enumerate(asins)}
        import json as _json2

        (out_dir / f"{prefix}_id_to_idx_qwen.json").write_text(
            _json2.dumps(id_to_idx, ensure_ascii=False)
        )
        (out_dir / f"{prefix}_texts.json").write_text(
            _json2.dumps(dict(zip(asins, texts)), ensure_ascii=False)
        )
        log.info(f"  Saved {prefix}_id_to_idx_qwen.json ({len(id_to_idx):,} entries)")

        ckpt = out_dir / f"{prefix}_embeddings_checkpoint_{args.model.replace('/', '_')}.npy"
        t0_domain = time.time()
        embeddings = embed_with_checkpoint(texts, model, ckpt, batch_size=args.batch_size)
        elapsed = time.time() - t0_domain

        out_path = out_dir / f"{prefix}_embeddings_qwen.npy"
        np.save(out_path, embeddings)
        log.info(f"  ✓ {out_path.name} — shape: {embeddings.shape} — {elapsed:.0f}s")

        stats[prefix] = {
            "text_format": text_format,
            "n_items": len(asins),
            "n_sparse_metadata": len(empty),
            "shape": list(embeddings.shape),
            "elapsed_seconds": round(elapsed, 1),
        }
        sanity_check_embeddings(embeddings)

        stats["total_elapsed_seconds"] = round(time.time() - t0, 1)
        stats_path = out_dir / f"{prefix}_embedding_stats.json"
        stats_path.write_text(_json2.dumps(stats, indent=2, ensure_ascii=False))
        log.info(f"\nDONE — {prefix} embeddings generated in {elapsed:.0f}s")
        return

    # ------------------------------------------------------------------
    # 2b. Books
    # ------------------------------------------------------------------
    if not args.albums_only:
        log.info("")
        log.info("─" * 65)
        log.info("BOOKS")
        log.info("─" * 65)

        books_path = RAW_DIR / "books_filtered.jsonl"
        if not books_path.exists():
            raise FileNotFoundError(
                f"Books file not found: {books_path}\n"
                "Make sure you have run src/acw/data/amazon_preprocessing.py first."
            )

        book_asins, book_texts, book_empty = load_jsonl(books_path, prepare_book_text, "Book")

        book_id_to_idx = {asin: idx for idx, asin in enumerate(book_asins)}
        (PROCESSED_DIR / "book_id_to_idx.json").write_text(
            json.dumps(book_id_to_idx, ensure_ascii=False)
        )
        (PROCESSED_DIR / "books_texts.json").write_text(json.dumps(book_texts, ensure_ascii=False))
        log.info(f"  Saved book_id_to_idx.json ({len(book_id_to_idx):,} entries)")

        n_books = len(book_texts)
        est_minutes = (n_books / args.batch_size) / (200 if device == "cuda" else 20)
        log.info(f"  Embedding {n_books:,} books (estimated ~{est_minutes:.0f} min on {device})...")

        t_books = time.time()
        book_ckpt = PROCESSED_DIR / f"book_embeddings_checkpoint_{args.model.replace('/', '_')}.npy"
        book_embeddings = embed_with_checkpoint(
            book_texts, model, book_ckpt, batch_size=args.batch_size
        )
        elapsed_books = time.time() - t_books

        out_path = PROCESSED_DIR / "book_embeddings.npy"
        np.save(out_path, book_embeddings)
        log.info(
            f"  ✓ book_embeddings.npy saved — shape: {book_embeddings.shape} — {elapsed_books:.0f}s"
        )

        stats["books"] = {
            "n_items": n_books,
            "n_sparse_metadata": len(book_empty),
            "shape": list(book_embeddings.shape),
            "elapsed_seconds": round(elapsed_books, 1),
        }

        sanity_check_embeddings(book_embeddings)

    # ------------------------------------------------------------------
    # 3. Albums
    # ------------------------------------------------------------------
    if not args.books_only:
        log.info("")
        log.info("─" * 65)
        log.info("ALBUMS")
        log.info("─" * 65)

        albums_path = RAW_DIR / "albums_filtered.jsonl"
        if not albums_path.exists():
            raise FileNotFoundError(
                f"Albums file not found: {albums_path}\n"
                "Make sure you have run src/acw/data/amazon_preprocessing.py first."
            )

        album_asins, album_texts, album_empty = load_jsonl(albums_path, prepare_album_text, "Album")

        album_id_to_idx = {asin: idx for idx, asin in enumerate(album_asins)}
        (PROCESSED_DIR / "album_id_to_idx.json").write_text(
            json.dumps(album_id_to_idx, ensure_ascii=False)
        )
        (PROCESSED_DIR / "albums_texts.json").write_text(
            json.dumps(album_texts, ensure_ascii=False)
        )
        log.info(f"  Saved album_id_to_idx.json ({len(album_id_to_idx):,} entries)")

        n_albums = len(album_texts)
        est_minutes = (n_albums / args.batch_size) / (200 if device == "cuda" else 20)
        log.info(
            f"  Embedding {n_albums:,} albums (estimated ~{est_minutes:.0f} min on {device})..."
        )

        t_albums = time.time()
        album_ckpt = (
            PROCESSED_DIR / f"album_embeddings_checkpoint_{args.model.replace('/', '_')}.npy"
        )
        album_embeddings = embed_with_checkpoint(
            album_texts, model, album_ckpt, batch_size=args.batch_size
        )
        elapsed_albums = time.time() - t_albums

        out_path = PROCESSED_DIR / "album_embeddings.npy"
        np.save(out_path, album_embeddings)
        log.info(
            f"  ✓ album_embeddings.npy saved — "
            f"shape: {album_embeddings.shape} — {elapsed_albums:.0f}s"
        )

        stats["albums"] = {
            "n_items": n_albums,
            "n_sparse_metadata": len(album_empty),
            "shape": list(album_embeddings.shape),
            "elapsed_seconds": round(elapsed_albums, 1),
        }

    # ------------------------------------------------------------------
    # 4. Save run stats
    # ------------------------------------------------------------------
    stats["total_elapsed_seconds"] = round(time.time() - t0, 1)
    stats_path = PROCESSED_DIR / "embedding_stats.json"
    stats_path.write_text(json.dumps(stats, indent=2, ensure_ascii=False))

    log.info("")
    log.info("=" * 65)
    log.info("DONE")
    log.info("=" * 65)
    if "books" in stats:
        log.info(f"  Books  : {stats['books']['shape']}  ({stats['books']['elapsed_seconds']}s)")
    if "albums" in stats:
        log.info(f"  Albums : {stats['albums']['shape']}  ({stats['albums']['elapsed_seconds']}s)")
    log.info(f"  Total  : {stats['total_elapsed_seconds']}s")
    log.info("")
    log.info("Next step:")
    log.info("  python -m acw.evaluation.evaluate --all")
    log.info("=" * 65)
