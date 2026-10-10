"""
experiments/encoder_sweep/bge_significance.py
=============================================
Significance and catalog-swap control for the BGE-large replication of
catalog whitening.

Part 1: significance tests (Wilcoxon-Pratt + permutation) for H2-BGE-raw
vs. H2-BGE-whitened on the music, movies and games domains, paired by user.

Part 2: book-fitted whitening control for BGE: fit ZCA on the book
embeddings (BGE) and apply it to each domain catalog, comparing the
resulting anisotropy (mean pairwise cosine) with the catalog's own
whitening.

Supports: the statistical significance of the BGE cross-encoder replication
and its encoder-level (catalog-swap) check.

Inputs: everything is read from data/processed/bge_revalidation/ (produced
        by experiments/encoder_sweep/bge_revalidation.py) plus the raw
        ground truth; no canonical artifact is touched.
Output: results are printed; Part 2 also writes
        data/processed/bge_revalidation/<domain>_catalog_whitened_offdomain_bge.npy.

Usage:
    python -m experiments.encoder_sweep.bge_significance
"""

import json

import numpy as np

from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users
from acw.evaluation.metrics import ndcg_at_k, precision_at_k
from acw.evaluation.significance_tests import run_pair
from acw.settings import ROOT  # noqa: E402
from acw.whitening import ZCAWhitening

PROC = ROOT / "data" / "processed"
OUT = PROC / "bge_revalidation"
RAW = ROOT / "data" / "raw"

K_VALUES = (5, 10, 20)


def per_user_scores(profiles, idx_map, items_emb, idx_to_item, genre_gt, test_users):
    n5, n10, p10 = [], [], []
    used_users = []
    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        row = idx_map.get(uid)
        if row is None:
            continue
        sims = items_emb @ profiles[row]
        top = np.argsort(sims)[::-1][:20]
        ranked = [idx_to_item[i] for i in top]
        n5.append(ndcg_at_k(ranked, relevant, 5))
        n10.append(ndcg_at_k(ranked, relevant, 10))
        p10.append(precision_at_k(ranked, relevant, 10))
        used_users.append(uid)
    return np.array(n5), np.array(n10), np.array(p10), used_users


def eval_domain(
    domain,
    items_emb_path,
    items_idx_path,
    items_meta_path,
    gt_path,
    canon_path,
    profiles_path,
    index_path,
):
    cfg = resolve_paths(
        domain=domain,
        gt_path=str(gt_path),
        items_emb_path=str(items_emb_path),
        items_idx_path=str(items_idx_path),
        canonical_users_path=str(canon_path),
        items_meta_path=str(items_meta_path),
    )
    items_emb = np.load(cfg.items_emb_path).astype(np.float32)
    item_id_to_idx = json.load(open(cfg.items_idx_path))
    idx_to_item = {v: k for k, v in item_id_to_idx.items()}
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)

    profiles = np.load(profiles_path).astype(np.float32)
    idx_map = json.load(open(index_path))

    n5, n10, p10, used = per_user_scores(
        profiles, idx_map, items_emb, idx_to_item, genre_gt, test_users
    )
    return n5, n10, p10, used


def part1_significance():
    print("=" * 70)
    print("PART 1 — Significance: H2-BGE-raw vs H2-BGE-whitened")
    print("=" * 70)

    domains = {
        "music": dict(
            gt_path=RAW / "ground_truth_real.json",
            canon_path=PROC / "canonical_test_users.json",
            meta_path=RAW / "albums_filtered.jsonl",
            raw_emb=OUT / "albums_embeddings_bge.npy",
            white_emb=OUT / "music_catalog_whitened_bge.npy",
            idx_path=OUT / "albums_id_to_idx_bge.json",
        ),
        "movies": dict(
            gt_path=RAW / "movies" / "ground_truth_real.json",
            canon_path=PROC / "canonical_test_users_movies.json",
            meta_path=RAW / "movies" / "movies_filtered.jsonl",
            raw_emb=OUT / "movies_embeddings_bge.npy",
            white_emb=OUT / "movies_catalog_whitened_bge.npy",
            idx_path=OUT / "movies_id_to_idx_bge.json",
        ),
        "games": dict(
            gt_path=RAW / "games" / "ground_truth_real.json",
            canon_path=PROC / "canonical_test_users_games.json",
            meta_path=RAW / "games" / "games_filtered.jsonl",
            raw_emb=OUT / "games_embeddings_bge.npy",
            white_emb=OUT / "games_catalog_whitened_bge.npy",
            idx_path=OUT / "games_id_to_idx_bge.json",
        ),
    }

    for domain, d in domains.items():
        prof_path = OUT / f"h2_{domain}_bge.npy"
        idx_path = OUT / f"h2_{domain}_bge_index.json"

        n5_raw, n10_raw, p10_raw, used_raw = eval_domain(
            domain,
            d["raw_emb"],
            d["idx_path"],
            d["meta_path"],
            d["gt_path"],
            d["canon_path"],
            prof_path,
            idx_path,
        )
        n5_w, n10_w, p10_w, used_w = eval_domain(
            domain,
            d["white_emb"],
            d["idx_path"],
            d["meta_path"],
            d["gt_path"],
            d["canon_path"],
            prof_path,
            idx_path,
        )

        assert used_raw == used_w, f"{domain}: user populations differ between raw and whitened"

        res = run_pair(
            pair_id=f"bge_{domain}",
            family="bge_revalidation",
            method_a="H2-BGE-whitened",
            method_b="H2-BGE-raw",
            desc=f"BGE catalog-whitening effect, {domain}",
            ndcg5_a=n5_w,
            ndcg10_a=n10_w,
            p10_a=p10_w,
            ndcg5_b=n5_raw,
            ndcg10_b=n10_raw,
            p10_b=p10_raw,
        )
        print(
            f"\n[{domain}] n={res['n_users_total']}  "
            f"mean_whitened={res['mean_ndcg10_a']:.4f}  mean_raw={res['mean_ndcg10_b']:.4f}  "
            f"delta={res['delta_ndcg10']:+.4f}"
        )
        print(
            f"  Wilcoxon p={res['p_wilcoxon_raw']:.4g}  Perm p={res['p_perm_raw']:.4g}  "
            f"effect_r={res['effect_size_r']:.3f} ({res['effect_size_label']})"
        )


def mean_cos_sim(X, sample=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    S = X[idx]
    sim = S @ S.T
    iu = np.triu_indices_from(sim, k=1)
    return float(sim[iu].mean())


def part2_m3_offdomain():
    print("\n" + "=" * 70)
    print("PART 2 — Book-fitted ZCA control (BGE): ZCA fitted on books vs. own catalog")
    print("=" * 70)

    book_emb = np.load(OUT / "books_embeddings_bge.npy").astype(np.float64)
    zca_books = ZCAWhitening().fit(book_emb)

    catalogs = {
        "music": OUT / "albums_embeddings_bge.npy",
        "movies": OUT / "movies_embeddings_bge.npy",
        "games": OUT / "games_embeddings_bge.npy",
    }
    own_whitened = {
        "music": OUT / "music_catalog_whitened_bge.npy",
        "movies": OUT / "movies_catalog_whitened_bge.npy",
        "games": OUT / "games_catalog_whitened_bge.npy",
    }

    for domain, path in catalogs.items():
        emb = np.load(path).astype(np.float32)
        offdomain_white = zca_books.transform(emb.astype(np.float64))
        own_white = np.load(own_whitened[domain])

        sim_raw = mean_cos_sim(emb)
        sim_offdomain = mean_cos_sim(offdomain_white)
        sim_own = mean_cos_sim(own_white)

        np.save(OUT / f"{domain}_catalog_whitened_offdomain_bge.npy", offdomain_white)

        print(
            f"\n[{domain}] raw_sim={sim_raw:.4f}  "
            f"own-whitened_sim={sim_own:.4f}  offdomain(books)-whitened_sim={sim_offdomain:.4f}"
        )


if __name__ == "__main__":
    part1_significance()
    part2_m3_offdomain()
