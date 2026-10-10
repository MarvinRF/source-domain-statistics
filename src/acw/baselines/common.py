"""Helpers shared by the learned alignment baselines.

Ground truth and ranking metrics come from ``acw.evaluation`` so that the
relevance used to train a module is exactly the one used to evaluate it.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_ground_truth
from acw.evaluation.pipeline import evaluate_method
from acw.settings import PROCESSED_DIR, RAW_DIR

log = logging.getLogger(__name__)

ALIGNMENT_DIR = PROCESSED_DIR / "alignment"
EPS = 1e-6


def domain_paths(domain: str) -> dict[str, Path]:
    """Item-side artifact paths for a target domain (profiles always come from books)."""
    if domain == "music":
        return {
            "items_emb_path": ALIGNMENT_DIR / "album_embeddings_whitened.npy",
            "items_id_map_path": PROCESSED_DIR / "album_id_to_idx.json",
            "gt_path": RAW_DIR / "ground_truth_real.json",
            "items_jsonl_path": RAW_DIR / "albums_filtered.jsonl",
            "domain_dir": ALIGNMENT_DIR,
        }
    domain_dir = PROCESSED_DIR / domain / "alignment"
    domain_dir.mkdir(parents=True, exist_ok=True)
    gt_path = RAW_DIR / domain / f"ground_truth_{domain}.json"
    if not gt_path.exists():
        gt_path = RAW_DIR / domain / "ground_truth_real.json"
    id_map = PROCESSED_DIR / domain / f"{domain}_id_to_idx.json"
    if not id_map.exists():
        id_map = PROCESSED_DIR / domain / f"{domain}_id_to_idx_qwen.json"
    return {
        "items_emb_path": domain_dir / f"{domain}_embeddings_whitened.npy",
        "items_id_map_path": id_map,
        "gt_path": gt_path,
        "items_jsonl_path": RAW_DIR / domain / f"{domain}_filtered.jsonl",
        "domain_dir": domain_dir,
    }


def ground_truth(
    domain: str = "music",
    gt_path: Path | None = None,
    items_jsonl_path: Path | None = None,
    items_id_map_path: Path | None = None,
    mode: str = "genre",
) -> dict[str, set[str]]:
    """Relevant items per user, built by ``acw.evaluation.loader.build_ground_truth``."""
    cfg = resolve_paths(
        domain,
        gt_path=str(gt_path) if gt_path else None,
        items_meta_path=str(items_jsonl_path) if items_jsonl_path else None,
        items_idx_path=str(items_id_map_path) if items_id_map_path else None,
    )
    return build_ground_truth(cfg, mode)


def apply_zca_book(profiles: np.ndarray) -> np.ndarray:
    """Whiten profiles with the books-catalog ZCA (the ``-ZB`` pre-stage).

    Requires ``data/processed/alignment/whitening_book.npz``, written by
    ``python -m acw.whitening --save-aligned``.
    """
    zca_path = ALIGNMENT_DIR / "whitening_book.npz"
    if not zca_path.exists():
        raise FileNotFoundError(
            f"ZCA-book params not found at {zca_path}. Run: python -m acw.whitening --save-aligned"
        )
    data = np.load(zca_path)
    xw = (profiles.astype(np.float64) - data["mean"]) @ data["W"].T
    norms = np.linalg.norm(xw, axis=1, keepdims=True)
    return (xw / np.maximum(norms, EPS)).astype(np.float32)


def compute_centroids(
    items_emb: np.ndarray,
    item_id_to_idx: dict[str, int],
    gt: dict[str, set[str]],
    user_ids: list[str],
) -> np.ndarray:
    """L2-normalized centroid of each user's relevant items (the InfoNCE positive)."""
    centroids = np.zeros((len(user_ids), items_emb.shape[1]), dtype=np.float32)
    for i, uid in enumerate(user_ids):
        idxs = [item_id_to_idx[a] for a in gt.get(uid, ()) if a in item_id_to_idx]
        if idxs:
            c = items_emb[idxs].mean(axis=0).astype(np.float64)
            centroids[i] = (c / max(np.linalg.norm(c), EPS)).astype(np.float32)
    log.info(
        f"  Centroids computed for {len(user_ids):,} users, non-zero: {centroids.any(axis=1).sum():,}"
    )
    return centroids


def evaluate(
    label: str,
    profiles: np.ndarray,
    items_emb: np.ndarray,
    idx_map: dict[str, int],
    idx_to_item: dict[int, str],
    gt: dict[str, set[str]],
) -> dict[str, float]:
    """Rank the catalog for every user in ``idx_map`` with the canonical evaluator."""
    return evaluate_method(label, profiles, items_emb, idx_map, idx_to_item, gt, list(idx_map))
