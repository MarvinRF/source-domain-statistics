"""
experiments/encoder_sweep/full_pipeline.py
==========================================
Full pipeline (catalog embedding, H2 profiles, ZCA whitening, evaluation,
significance) for the candidate backbones selected after the eigenvalue
spectrum screening: all-MiniLM-L6-v2 and nomic-embed-text-v1.5.

Protocol identical to the one used for GTE/BGE/Qwen3: music domain,
canonical_val_users.json population (n=2,731), MAX_BOOKS_PROFILE=10,
full-rank ZCA. `run_backbone` is reused by full_pipeline_round2.py.

Supports: the encoder sweep relating embedding anisotropy to the
catalog-whitening gain (including encoders where whitening hurts).

Inputs:  data/raw/{books,albums}_filtered.jsonl, user_book_history.json,
         data/processed/canonical_val_users.json
Outputs: data/processed/backbone_screening/full_pipeline_results.json plus
         per-backbone embeddings and whitened catalogs in the same directory
         (nothing is written outside it).

Usage:
    python -m experiments.encoder_sweep.full_pipeline
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users
from acw.evaluation.metrics import ndcg_at_k, precision_at_k
from acw.evaluation.significance_tests import run_pair
from acw.settings import ROOT  # noqa: E402
from acw.whitening import ZCAWhitening

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = PROC / "backbone_screening"
OUT.mkdir(parents=True, exist_ok=True)

MAX_BOOKS_PROFILE = 10


def log(msg):
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl_texts(path, id_field="parent_asin"):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d[id_field])
            texts.append(d["text"])
    return ids, texts


def embed_catalog(model, tag, name, ids, texts, prefix="", batch_size=64):
    emb_path = OUT / f"{tag}_embeddings_{name}.npy"
    idx_path = OUT / f"{tag}_id_to_idx_{name}.json"
    if emb_path.exists() and idx_path.exists():
        log(f"{tag} ({name}): already exists, skipping.")
        return np.load(emb_path), json.load(open(idx_path))

    full_texts = [prefix + t for t in texts] if prefix else texts
    log(f"{tag} ({name}): embedding {len(full_texts):,} items (batch_size={batch_size})...")
    emb = model.encode(
        full_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    id_to_idx = {iid: i for i, iid in enumerate(ids)}
    np.save(emb_path, emb)
    json.dump(id_to_idx, open(idx_path, "w"))
    log(f"{tag} ({name}): done, shape={emb.shape}")
    return emb, id_to_idx


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


def per_user(profiles, idx_map, items_emb, idx_to_item, genre_gt, test_users):
    n5, n10, p10, used = [], [], [], []
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
        used.append(uid)
    return np.array(n5), np.array(n10), np.array(p10), used


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


def run_backbone(model_id, name, prefix="", batch_size=64):
    log(f"\n{'=' * 70}\n{name}\n{'=' * 70}")
    model = SentenceTransformer(model_id, device="cuda", trust_remote_code=True)

    book_ids, book_texts = load_jsonl_texts(RAW / "books_filtered.jsonl")
    album_ids, album_texts = load_jsonl_texts(RAW / "albums_filtered.jsonl")

    book_emb, book_id_to_idx = embed_catalog(
        model, "books", name, book_ids, book_texts, prefix, batch_size=batch_size
    )
    album_emb, album_id_to_idx = embed_catalog(
        model, "albums", name, album_ids, album_texts, prefix, batch_size=batch_size
    )
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}

    history = json.load(open(RAW / "user_book_history.json"))
    val_users = json.load(open(PROC / "canonical_val_users.json"))
    h2_profiles, valid_users = build_h2(book_emb, book_id_to_idx, history, val_users)
    idx_map = {u: i for i, u in enumerate(valid_users)}
    log(f"H2: {len(valid_users):,}/{len(val_users):,} valid users")

    raw_sim = mean_cos_sim(album_emb)
    zca = ZCAWhitening().fit(album_emb.astype(np.float64))
    whitened = zca.transform(album_emb.astype(np.float64))
    white_sim = mean_cos_sim(whitened)
    np.save(OUT / f"music_catalog_whitened_{name}.npy", whitened)
    log(f"raw_sim={raw_sim:.4f}  whitened_sim={white_sim:.4f}")

    cfg = resolve_paths(
        domain="music",
        gt_path=str(RAW / "ground_truth_real.json"),
        items_emb_path=str(OUT / f"albums_embeddings_{name}.npy"),
        items_idx_path=str(OUT / f"albums_id_to_idx_{name}.json"),
        items_meta_path=str(RAW / "albums_filtered.jsonl"),
    )
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)

    n5r, n10r, p10r, ur = per_user(
        h2_profiles, idx_map, album_emb, idx_to_album, genre_gt, test_users
    )
    n5w, n10w, p10w, uw = per_user(
        h2_profiles, idx_map, whitened, idx_to_album, genre_gt, test_users
    )
    assert ur == uw
    rnd = random_baseline(genre_gt, test_users, idx_to_album, len(idx_to_album))

    res = run_pair(
        pair_id=f"{name}_music",
        family="full_pipeline_candidates",
        method_a=f"H2-{name}-whitened",
        method_b=f"H2-{name}-raw",
        desc=f"{name} catalog-whitening, music, n=2731",
        ndcg5_a=n5w,
        ndcg10_a=n10w,
        p10_a=p10w,
        ndcg5_b=n5r,
        ndcg10_b=n10r,
        p10_b=p10r,
    )
    log(
        f"Random={rnd:.4f}  raw={res['mean_ndcg10_b']:.4f}  whitened={res['mean_ndcg10_a']:.4f}  "
        f"delta={res['delta_ndcg10']:+.4f}  p_wilcoxon={res['p_wilcoxon_raw']:.4g}  "
        f"p_perm={res['p_perm_raw']:.4g}  effect_r={res['effect_size_r']:.3f} ({res['effect_size_label']})"
    )

    return {
        "random": rnd,
        "raw": res["mean_ndcg10_b"],
        "whitened": res["mean_ndcg10_a"],
        "delta": res["delta_ndcg10"],
        "p_wilcoxon": res["p_wilcoxon_raw"],
        "p_perm": res["p_perm_raw"],
        "effect_r": res["effect_size_r"],
        "effect_label": res["effect_size_label"],
        "n": res["n_users_total"],
        "raw_sim": raw_sim,
        "whitened_sim": white_sim,
    }


def main():
    results = {}
    results["minilm"] = run_backbone("sentence-transformers/all-MiniLM-L6-v2", "minilm")
    results["nomic"] = run_backbone(
        "nomic-ai/nomic-embed-text-v1.5", "nomic", prefix="search_document: "
    )

    json.dump(results, open(OUT / "full_pipeline_results.json", "w"), indent=2)
    log("\nSaved to backbone_screening/full_pipeline_results.json")


if __name__ == "__main__":
    main()
