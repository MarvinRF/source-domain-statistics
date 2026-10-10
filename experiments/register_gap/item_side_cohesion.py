"""
experiments/register_gap/item_side_cohesion.py
==============================================
Item-side semantic-cohesion diagnostic ("Album-SA").

Measures the intra-subgenre semantic cohesion of the album embeddings,
following the SA-Init motivation of TS-Rec (arXiv:2602.22632).

Steps:
1. Parse albums_filtered.jsonl -> subgenre per album (same rule as the
   evaluation pipeline: cats[min(2, len(cats)-1)]).
2. Keep subgenres with >= MIN_GENRE_SIZE albums that have an embedding.
3. For each subgenre, compute the mean pairwise cosine similarity
   (vectorised, no per-pair loop).
4. Report mean, median, min, max and the % of subgenres with avg_sim < 0.5.

Decision rule (stored as "go_tarefa3" in the output): an item-side
re-weighting is warranted only if avg_sim < 0.5 in more than 50% of the
subgenres.

Supports: the item-side analysis of the register gap (whether the catalog
side is already semantically cohesive).

Inputs: data/processed/album_embeddings.npy, album_id_to_idx.json,
        data/raw/albums_filtered.jsonl
Output: results/evaluation/album_sa_diagnosis_{timestamp}.json

Usage:
    python -m experiments.register_gap.item_side_cohesion
"""

from __future__ import annotations

import json
import logging
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results" / "evaluation"

MIN_GENRE_SIZE = 5


def main() -> None:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    RESULTS.mkdir(parents=True, exist_ok=True)

    log.info("=" * 65)
    log.info("Album-SA Diagnosis — intra-subgenre cohesion")
    log.info("=" * 65)

    # ── Load embeddings ───────────────────────────────────────────────────────
    log.info("\nLoading album_embeddings.npy...")
    album_emb = np.load(PROCESSED / "album_embeddings.npy").astype(np.float32)
    with open(PROCESSED / "album_id_to_idx.json") as f:
        album_id_to_idx: dict[str, int] = json.load(f)
    log.info(f"  Albums: {album_emb.shape}")

    # ── Parse subgenres ───────────────────────────────────────────────────────
    log.info("\nParsing albums_filtered.jsonl...")
    genre_to_asins: dict[str, list[str]] = defaultdict(list)

    with open(RAW / "albums_filtered.jsonl") as f:
        for line in f:
            item = json.loads(line)
            asin = item.get("parent_asin")
            cats: list[str] = item.get("categories", [])
            if not asin or len(cats) < 1:
                continue
            genre = cats[min(2, len(cats) - 1)]
            genre_to_asins[genre].append(asin)

    total_genres = len(genre_to_asins)
    log.info(f"  Subgenres found: {total_genres}")

    # Keep subgenres with >= MIN_GENRE_SIZE albums that have an embedding
    filtered: dict[str, list[int]] = {}
    for genre, asins in genre_to_asins.items():
        idxs = [album_id_to_idx[a] for a in asins if a in album_id_to_idx]
        if len(idxs) >= MIN_GENRE_SIZE:
            filtered[genre] = idxs

    log.info(f"  Subgenres with >= {MIN_GENRE_SIZE} albums: {len(filtered)}")

    # ── Intra-subgenre avg_sim ────────────────────────────────────────────────
    log.info("\nComputing intra-subgenre avg_sim (vectorised)...")

    genre_stats: list[dict] = []

    for genre, idxs in filtered.items():
        M = album_emb[idxs]  # (n, 1024), already L2-normalised
        n = len(M)
        # Matrix product: (n, n) cosine similarities
        sim_matrix = M @ M.T  # (n, n)
        # Sum excluding the diagonal (self-similarities = 1.0)
        total_sim = float(sim_matrix.sum()) - n
        avg_sim = total_sim / (n * (n - 1)) if n > 1 else 0.0
        genre_stats.append({"genre": genre, "n": n, "avg_sim": avg_sim})

    # ── Global statistics ─────────────────────────────────────────────────────
    sims = np.array([g["avg_sim"] for g in genre_stats], dtype=np.float32)
    pct_below_05 = float((sims < 0.5).mean()) * 100.0

    summary = {
        "n_genres_total": total_genres,
        "n_genres_filtered": len(filtered),
        "min_genre_size": MIN_GENRE_SIZE,
        "avg_sim_mean": float(sims.mean()),
        "avg_sim_median": float(np.median(sims)),
        "avg_sim_min": float(sims.min()),
        "avg_sim_max": float(sims.max()),
        "avg_sim_std": float(sims.std()),
        "pct_below_0.5": pct_below_05,
        "go_tarefa3": pct_below_05 > 50.0,
        "per_genre": sorted(genre_stats, key=lambda x: x["avg_sim"]),
    }

    # ── Log summary ───────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("Album-SA Diagnosis — Results")
    log.info("=" * 65)
    log.info(f"  Subgenres evaluated  : {len(filtered)}")
    log.info(f"  avg_sim mean         : {summary['avg_sim_mean']:.4f}")
    log.info(f"  avg_sim median       : {summary['avg_sim_median']:.4f}")
    log.info(f"  avg_sim min          : {summary['avg_sim_min']:.4f}")
    log.info(f"  avg_sim max          : {summary['avg_sim_max']:.4f}")
    log.info(f"  % subgenres < 0.5    : {pct_below_05:.1f}%")
    log.info(f"  GO for re-weighting  : {summary['go_tarefa3']}")
    log.info("=" * 65)

    log.info("\nTop 10 least cohesive subgenres:")
    for g in summary["per_genre"][:10]:
        log.info(f"  {g['genre'][:50]:<50}  n={g['n']:>5}  avg_sim={g['avg_sim']:.4f}")

    log.info("\nTop 10 most cohesive subgenres:")
    for g in summary["per_genre"][-10:][::-1]:
        log.info(f"  {g['genre'][:50]:<50}  n={g['n']:>5}  avg_sim={g['avg_sim']:.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = RESULTS / f"album_sa_diagnosis_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log.info(f"\nDiagnostic saved: {out_path}")

    if summary["go_tarefa3"]:
        log.info("→ Album-SA re-weighting: GO")
    else:
        log.info("→ Album-SA re-weighting: SKIP — intra-subgenre cohesion already adequate")

    log.info("\nDONE")


if __name__ == "__main__":
    main()
