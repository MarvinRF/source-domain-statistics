"""
experiments/robustness/truncated_whitening.py
=============================================
Tests whether truncated whitening (ZCA restricted to the top-k principal
components instead of full rank) recovers the catalog-whitening gain on
backbones that collapse under full-rank ZCA (Qwen3-Embedding-8B, BGE-large),
on the music domain. Reports H2 nDCG@10 and the mean pairwise cosine of the
whitened catalog for each truncation level k, alongside the raw and Random
baselines.

Supports: the failure-mode analysis of catalog whitening on extremely
anisotropic encoders (whether a rank-truncated variant rescues them).

Inputs: data/processed/{book,album}_embeddings_qwen3.npy and
        data/processed/bge_revalidation/{books,albums}_embeddings_bge.npy
        (produced by experiments/encoder_sweep/bge_revalidation.py).
Output: data/processed/bge_revalidation/truncated_whitening_results.json
        (no canonical artifact is modified).

Usage:
    python -m experiments.robustness.truncated_whitening
"""

import json

import numpy as np

from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users
from acw.evaluation.metrics import ndcg_at_k
from acw.settings import ROOT  # noqa: E402

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = PROC / "bge_revalidation"
MAX_BOOKS_PROFILE = 10


def mean_cos_sim(X, sample=5000, seed=42):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    S = X[idx]
    sim = S @ S.T
    iu = np.triu_indices_from(sim, k=1)
    return float(sim[iu].mean())


def build_h2(book_emb, book_id_to_idx, history, test_users):
    profiles, valid = [], []
    for uid in test_users:
        books = history.get(uid, [])[-MAX_BOOKS_PROFILE:]
        idxs = [book_id_to_idx[b] for b in books if b in book_id_to_idx]
        if not idxs:
            continue
        v = book_emb[idxs].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        profiles.append(v.astype(np.float32))
        valid.append(uid)
    return np.array(profiles, dtype=np.float32), valid


def evaluate_h2(profiles, idx_map, items_emb, idx_to_item, genre_gt, test_users):
    n10 = []
    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        row = idx_map.get(uid)
        if row is None:
            continue
        sims = items_emb @ profiles[row]
        top = np.argsort(sims)[::-1][:10]
        ranked = [idx_to_item[i] for i in top]
        n10.append(ndcg_at_k(ranked, relevant, 10))
    return float(np.mean(n10)), len(n10)


def random_baseline(genre_gt, test_users, idx_to_item, n_items, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        idxs = rng.choice(n_items, size=10, replace=False)
        ranked = [idx_to_item[i] for i in idxs]
        scores.append(ndcg_at_k(ranked, relevant, 10))
    return float(np.mean(scores))


def run_backbone(name, book_emb_path, album_emb_path, k_values):
    print(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    book_emb = np.load(book_emb_path).astype(np.float32)
    album_emb = np.load(album_emb_path).astype(np.float32)
    book_id_to_idx = json.load(open(PROC / "book_id_to_idx.json"))
    album_id_to_idx = json.load(open(PROC / "album_id_to_idx.json"))
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}

    history = json.load(open(RAW / "user_book_history.json"))
    test_users_canon = json.load(open(PROC / "canonical_test_users.json"))
    h2_profiles, valid_users = build_h2(book_emb, book_id_to_idx, history, test_users_canon)
    idx_map = {u: i for i, u in enumerate(valid_users)}

    cfg = resolve_paths(
        domain="music",
        gt_path=str(RAW / "ground_truth_real.json"),
        items_emb_path=str(album_emb_path),
        items_idx_path=str(PROC / "album_id_to_idx.json"),
        canonical_users_path=str(PROC / "canonical_test_users.json"),
        items_meta_path=str(RAW / "albums_filtered.jsonl"),
    )
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)

    raw_norm = album_emb / (np.linalg.norm(album_emb, axis=1, keepdims=True) + 1e-12)
    ndcg_raw, n_raw = evaluate_h2(
        h2_profiles, idx_map, raw_norm, idx_to_album, genre_gt, test_users
    )
    rnd = random_baseline(genre_gt, test_users, idx_to_album, len(idx_to_album))
    raw_sim = mean_cos_sim(raw_norm)
    print(f"Random nDCG@10={rnd:.4f}")
    print(f"H2-raw nDCG@10={ndcg_raw:.4f} (n={n_raw}, raw_mean_sim={raw_sim:.4f})")

    results = {
        "random": rnd,
        "raw": ndcg_raw,
        "raw_mean_sim": raw_sim,
        "n_eval": n_raw,
        "truncated": {},
    }

    # Eigendecomposition computed once, reused for every truncation level.
    X = album_emb.astype(np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    cov = np.cov(Xc.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    eps = 1e-6

    for k in k_values:
        ev_k = eigenvalues[:k]
        V_k = eigenvectors[:, :k]
        D_inv_sqrt = np.diag(1.0 / np.sqrt(ev_k + eps))
        W = V_k @ D_inv_sqrt @ V_k.T
        whitened = Xc @ W.T
        norms = np.linalg.norm(whitened, axis=1, keepdims=True)
        whitened = (whitened / np.maximum(norms, eps)).astype(np.float32)

        sim = mean_cos_sim(whitened)
        ndcg, n_eval = evaluate_h2(
            h2_profiles, idx_map, whitened, idx_to_album, genre_gt, test_users
        )
        results["truncated"][k] = {"ndcg10": ndcg, "mean_sim": sim, "n_eval": n_eval}
        print(f"  k={k:>5}: nDCG@10={ndcg:.4f}  mean_sim={sim:.4f}  (n={n_eval})")

    return results


def main():
    all_results = {}

    all_results["qwen3"] = run_backbone(
        "Qwen3-Embedding-8B (4096d) — native effective rank: 5",
        PROC / "book_embeddings_qwen3.npy",
        PROC / "album_embeddings_qwen3.npy",
        k_values=[5, 10, 25, 50, 100, 250, 400, 500, 600, 750, 1000, 1500, 2000, 4096],
    )

    all_results["bge"] = run_backbone(
        "BGE-large-en-v1.5 (1024d) — native effective rank: 3",
        OUT / "books_embeddings_bge.npy",
        OUT / "albums_embeddings_bge.npy",
        k_values=[3, 5, 10, 25, 50, 100, 250, 500, 1024],
    )

    json.dump(all_results, open(OUT / "truncated_whitening_results.json", "w"), indent=2)
    print("\nSaved to bge_revalidation/truncated_whitening_results.json")


if __name__ == "__main__":
    main()
