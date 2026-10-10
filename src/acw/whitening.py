"""ZCA whitening of the source (books) and target (items) embedding spaces.

Fits an independent ZCA transform on each catalog: it centers the embeddings,
decorrelates their dimensions and rescales them to unit variance, which removes
the anisotropy of sentence-encoder spaces (a few dominant directions that
inflate cosine similarity and cause hubness). The book-side transform is applied
to the H2/H3/H4 user profiles and the item-side transform to the target catalog,
so retrieval compares both in whitened, L2-normalized coordinates. No training
is involved. When intent-entity caches are available, a diagnostic reports the
mean cosine similarity of book/album pairs that share the same
(atmosfera, estetica, contexto) entities before and after whitening.

Outputs go to ``data/processed/alignment/``: ``whitening_book.npz`` and
``user_profiles_{h2,h3,h4}_whitened.npy``, plus the item-side ZCA parameters and
whitened item embeddings (in ``data/processed/<domain>/alignment/`` for
non-music domains).

Usage::

    python -m acw.whitening                    # music, saves artifacts
    python -m acw.whitening --domain movies
    python -m acw.whitening --dry-run          # fit on a subsample, save nothing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
ALIGNMENT_DIR = PROCESSED / "alignment"
RESULTS_DIR = ROOT / "results" / "phase4"

ALIGNMENT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

SEED = 42
EPS = 1e-6  # numerical stability for whitening

# ─────────────────────────────────────────────────────────────────────────────
# ZCA Whitening
# ─────────────────────────────────────────────────────────────────────────────


class ZCAWhitening:
    """
    Zero-phase Component Analysis (ZCA) whitening.

    Transforms embeddings so that:
        - Mean is zero
        - Covariance is identity (decorrelated, unit variance)
        - Structure is maximally preserved (unlike PCA, ZCA minimizes
          the L2 distance between input and output)

    This is particularly effective for sentence-encoder embeddings, which
    are typically anisotropic (a few dimensions dominate cosine similarity,
    causing hubness and degraded retrieval).
    """

    def __init__(self, eps: float = EPS, n_components: int | None = None):
        self.eps = eps
        self.n_components = n_components  # None = keep all dimensions
        self.mean_: np.ndarray | None = None
        self.W_: np.ndarray | None = None  # whitening matrix
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> ZCAWhitening:
        """
        Fit ZCA whitening on embedding matrix X of shape (N, D).

        Steps:
            1. Center (subtract mean)
            2. Compute covariance matrix
            3. Eigendecompose covariance
            4. Build whitening matrix W = V @ diag(1/sqrt(λ+ε)) @ V^T
        """
        log.info(f"  Fitting ZCA on {X.shape[0]:,} × {X.shape[1]} embeddings...")
        t0 = time.time()

        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_

        # Covariance — use float64 for numerical stability
        cov = np.cov(X_centered.T.astype(np.float64))

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort descending
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Optional: keep only top-k components
        if self.n_components is not None:
            eigenvalues = eigenvalues[: self.n_components]
            eigenvectors = eigenvectors[:, : self.n_components]

        # Whitening matrix: W = V @ diag(1/sqrt(λ+ε)) @ V^T
        D_inv_sqrt = np.diag(1.0 / np.sqrt(eigenvalues + self.eps))
        self.W_ = eigenvectors @ D_inv_sqrt @ eigenvectors.T

        self.fitted_ = True
        log.info(f"  ZCA fit complete in {time.time() - t0:.1f}s")
        log.info(f"  Eigenvalue range: [{eigenvalues.min():.4f}, {eigenvalues.max():.4f}]")
        log.info(f"  Effective rank (λ > 0.01): {(eigenvalues > 0.01).sum()}")

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Apply whitening: X_white = (X - mean) @ W^T, then L2-normalize."""
        assert self.fitted_, "Call fit() before transform()"
        X_centered = X.astype(np.float64) - self.mean_
        X_white = X_centered @ self.W_.T
        # L2 normalize each vector
        norms = np.linalg.norm(X_white, axis=1, keepdims=True)
        norms = np.maximum(norms, EPS)
        return (X_white / norms).astype(np.float32)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def save(self, path: Path):
        np.savez(path, mean=self.mean_, W=self.W_)
        log.info(f"  ZCA params saved: {path.name}")

    @classmethod
    def load(cls, path: Path) -> ZCAWhitening:
        data = np.load(path)
        w = cls()
        w.mean_ = data["mean"]
        w.W_ = data["W"]
        w.fitted_ = True
        return w


# ─────────────────────────────────────────────────────────────────────────────
# Pair construction from intent entities
# ─────────────────────────────────────────────────────────────────────────────


def build_entity_pairs(
    book_intent: dict,
    album_intent: dict,
    max_pairs: int = 50_000,
) -> tuple[np.ndarray, np.ndarray, list[tuple]]:
    """
    Find (book_asin, album_asin) pairs with matching intent entities.

    Matching criterion: same (atmosfera, estetica, contexto) tuple.
    Rhythm/energy is excluded because book ``ritmo`` and album ``energia``
    use slightly different value ranges.

    Returns:
        book_asins  — list of book ASINs in pairs
        album_asins — list of album ASINs in pairs
        entity_keys — list of matching entity tuples (for diagnostics)
    """
    log.info("Building entity-based alignment pairs...")

    # Index books by entity key
    book_by_entity: dict[tuple, list[str]] = defaultdict(list)
    for asin, ent in book_intent.items():
        if ent.get("_fallback"):
            continue
        key = (
            ent.get("atmosfera", ""),
            ent.get("estetica", ""),
            ent.get("contexto", ""),
        )
        if all(key):
            book_by_entity[key].append(asin)

    # Index albums by entity key
    album_by_entity: dict[tuple, list[str]] = defaultdict(list)
    for asin, ent in album_intent.items():
        if ent.get("_fallback"):
            continue
        key = (
            ent.get("atmosfera", ""),
            ent.get("estetica", ""),
            ent.get("contexto", ""),
        )
        if all(key):
            album_by_entity[key].append(asin)

    # Find common keys
    common_keys = set(book_by_entity.keys()) & set(album_by_entity.keys())
    log.info(f"  Book entity keys  : {len(book_by_entity):,}")
    log.info(f"  Album entity keys : {len(album_by_entity):,}")
    log.info(f"  Common keys       : {len(common_keys):,}")

    # Build pairs — one book and one album per key (deterministic)
    rng = np.random.default_rng(SEED)
    book_asins = []
    album_asins = []
    entity_keys = []

    for key in sorted(common_keys):
        books = book_by_entity[key]
        albums = album_by_entity[key]
        # Sample one of each to avoid over-representing large clusters
        b = books[rng.integers(len(books))]
        a = albums[rng.integers(len(albums))]
        book_asins.append(b)
        album_asins.append(a)
        entity_keys.append(key)

        if len(book_asins) >= max_pairs:
            break

    log.info(f"  Pairs constructed : {len(book_asins):,}")
    return book_asins, album_asins, entity_keys


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline
# ─────────────────────────────────────────────────────────────────────────────


def main(dry_run: bool = False, save_aligned: bool = True, domain: str = "music"):
    log.info("=" * 70)
    log.info("ZCA Whitening Alignment")
    log.info(f"  Domain: {domain}")
    log.info("=" * 70)

    # ── Resolve item embedding path (domain-aware) ────────────────────────
    if domain == "music":
        items_emb_path = PROCESSED / "album_embeddings.npy"
        items_id_map_key = "album_id_to_idx"
        items_whitened_out = ALIGNMENT_DIR / "album_embeddings_whitened.npy"
        zca_items_out = ALIGNMENT_DIR / "whitening_album.npz"
    else:
        domain_dir = PROCESSED / domain
        # accept both {domain}_embeddings_qwen.npy and {domain}_embeddings.npy
        for candidate in [
            domain_dir / f"{domain}_embeddings_qwen.npy",
            domain_dir / f"{domain}_embeddings.npy",
        ]:
            if candidate.exists():
                items_emb_path = candidate
                break
        else:
            raise FileNotFoundError(
                f"Item embeddings not found for domain={domain} in {domain_dir}"
            )
        items_id_map_key = f"{domain}_id_to_idx"
        domain_align_dir = domain_dir / "alignment"
        domain_align_dir.mkdir(parents=True, exist_ok=True)
        items_whitened_out = domain_align_dir / f"{domain}_embeddings_whitened.npy"
        zca_items_out = domain_align_dir / f"whitening_{domain}.npz"

    # ── Load embeddings ───────────────────────────────────────────────────
    log.info("\nLoading embeddings...")
    items_emb = np.load(items_emb_path)
    log.info(f"  Items ({domain}): {items_emb.shape}")

    book_emb_path = PROCESSED / "book_embeddings.npy"
    if book_emb_path.exists():
        book_emb = np.load(book_emb_path)
        log.info(f"  Books : {book_emb.shape}")
        has_book_emb = True
    else:
        log.warning(
            "  book_embeddings.npy not found — ZCA-book will be fit on user profiles (H3/H4)"
        )
        book_emb = None
        has_book_emb = False

    # ── Load intent caches (optional — skip pair diagnostic if absent) ────
    book_intent_path = PROCESSED / "book_intent_cache.json"
    items_intent_path = PROCESSED / "album_intent_cache.json"
    has_intents = book_intent_path.exists() and items_intent_path.exists()
    if has_intents:
        log.info("\nLoading intent entity caches...")
        with open(book_intent_path) as f:
            book_intent = json.load(f)
        with open(items_intent_path) as f:
            album_intent = json.load(f)
        log.info(f"  Book entities : {len(book_intent):,}")
        log.info(f"  Album entities: {len(album_intent):,}")
    else:
        log.info("\nIntent caches not found — skipping pair alignment diagnostic.")
        book_intent = album_intent = None

    # ── Fit ZCA whitening ─────────────────────────────────────────────────
    log.info("\nFitting ZCA whitening...")
    rng = np.random.default_rng(SEED)

    if has_book_emb:
        sample_size = 50_000 if dry_run else len(book_emb)
        log.info(f"  Book whitening (fitting on {min(sample_size, len(book_emb)):,} samples)...")
        if sample_size < len(book_emb):
            idx_sample = rng.choice(len(book_emb), sample_size, replace=False)
            book_sample = book_emb[idx_sample]
        else:
            book_sample = book_emb
        zca_book = ZCAWhitening(eps=EPS)
        zca_book.fit(book_sample)
    else:
        zca_book = None

    log.info(f"  Item whitening (fitting on all {len(items_emb):,} items)...")
    zca_items = ZCAWhitening(eps=EPS)
    zca_items.fit(items_emb)

    # ── Transform item embeddings ─────────────────────────────────────────
    log.info("\nTransforming embeddings...")
    items_emb_w = zca_items.transform(items_emb)
    log.info(f"  Items whitened: {items_emb_w.shape}")

    # Pair diagnostic (only when intent caches and book embeddings are available)
    if has_intents and has_book_emb:
        with open(PROCESSED / "book_id_to_idx.json") as f:
            book_id_to_idx = json.load(f)
        # try both {domain}_id_to_idx.json and {domain}_id_to_idx_qwen.json
        for cand in [
            PROCESSED / f"{items_id_map_key}.json",
            PROCESSED / f"{items_id_map_key}_qwen.json",
            items_emb_path.parent / f"{items_id_map_key}.json",
            items_emb_path.parent / f"{items_id_map_key}_qwen.json",
        ]:
            if cand.exists():
                with open(cand) as f:
                    album_id_to_idx = json.load(f)
                break
        else:
            album_id_to_idx = {}

        book_pair_asins, album_pair_asins, _ = build_entity_pairs(
            book_intent,
            album_intent,
            max_pairs=5_000 if dry_run else 50_000,
        )
        book_pair_embs = np.array(
            [book_emb[book_id_to_idx[a]] for a in book_pair_asins if a in book_id_to_idx],
            dtype=np.float32,
        )
        album_pair_embs = np.array(
            [items_emb[album_id_to_idx[a]] for a in album_pair_asins if a in album_id_to_idx],
            dtype=np.float32,
        )
        if len(book_pair_embs) and len(album_pair_embs):
            book_pairs_w = zca_book.transform(book_pair_embs)
            album_pairs_w = zca_items.transform(album_pair_embs)
            sim_before = np.einsum("ij,ij->i", book_pair_embs, album_pair_embs).mean()
            sim_after = np.einsum("ij,ij->i", book_pairs_w, album_pairs_w).mean()
            log.info("\n  Pair alignment diagnostic:")
            log.info(f"    Mean cosine sim (before): {sim_before:.4f}")
            log.info(f"    Mean cosine sim (after) : {sim_after:.4f}")
            log.info(f"    Δ sim: {sim_after - sim_before:+.4f}")

    # ── Load user profiles and transform ──────────────────────────────────
    # Profiles live in the book space and are shared across target domains.
    log.info("\nTransforming user profiles...")

    # Build ZCA-book from profiles if book_embeddings.npy is absent
    if zca_book is None:
        # Fit on all available profile embeddings as proxy for book space
        all_profiles = []
        for h in ["h3", "h4", "h2"]:
            p = PROCESSED / f"user_profiles_{h}.npy"
            if p.exists():
                all_profiles.append(np.load(p))
        if all_profiles:
            proxy = np.concatenate(all_profiles, axis=0)
            log.info(f"  Fitting ZCA-book on {len(proxy):,} user profile vectors (proxy)...")
            zca_book = ZCAWhitening(eps=EPS)
            zca_book.fit(proxy)
        else:
            log.warning("  No profiles found — skipping profile whitening.")

    heuristics = ["h2", "h3", "h4"]
    profiles_whitened = {}
    profile_indices = {}

    if zca_book is not None:
        for h in heuristics:
            npy_path = PROCESSED / f"user_profiles_{h}.npy"
            index_path = PROCESSED / f"user_profile_index_{h}.json"
            if not npy_path.exists():
                log.warning(f"  {h.upper()}: missing, skipping")
                continue
            profiles = np.load(npy_path)
            with open(index_path) as f:
                idx_map = json.load(f)
            profiles_w = zca_book.transform(profiles)
            profiles_whitened[h] = profiles_w
            profile_indices[h] = idx_map
            log.info(f"  {h.upper()} whitened: {profiles_w.shape}")

    # ── Save artifacts ────────────────────────────────────────────────────
    if save_aligned and not dry_run:
        log.info("\nSaving aligned embeddings...")
        if zca_book is not None:
            zca_book.save(ALIGNMENT_DIR / "whitening_book.npz")
        zca_items.save(zca_items_out)

        np.save(items_whitened_out, items_emb_w)
        log.info(f"  Saved: {items_whitened_out}  {items_emb_w.shape}")
        for h, emb_w in profiles_whitened.items():
            np.save(ALIGNMENT_DIR / f"user_profiles_{h}_whitened.npy", emb_w)
            log.info(f"  Saved: user_profiles_{h}_whitened.npy  {emb_w.shape}")

    log.info("\n" + "=" * 70)
    log.info("DONE")
    log.info("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ZCA whitening of the book and item embedding spaces")
    p.add_argument("--dry-run", action="store_true", help="Fit on subset, skip saving")
    p.add_argument(
        "--no-save", action="store_true", help="Evaluate but do not save aligned embeddings"
    )
    p.add_argument(
        "--domain",
        default="music",
        choices=["music", "movies", "games"],
        help="Target domain for item embedding whitening (default: music)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(dry_run=args.dry_run, save_aligned=not args.no_save, domain=args.domain)
