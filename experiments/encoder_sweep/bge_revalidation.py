"""
experiments/encoder_sweep/bge_revalidation.py
=============================================
Standalone re-validation of the central finding (catalog-side whitening)
with BAAI/bge-large-en-v1.5 as a third backbone alongside GTE-large and
Qwen3-8B. Embeds the book, album, movie and game catalogs, fits ZCA per
domain catalog, records anisotropy diagnostics (mean pairwise cosine before
and after whitening, effective rank, condition number) and builds H2 user
profiles per domain.

Supports: the cross-encoder replication of the catalog-whitening gain and
the encoder sweep relating anisotropy to that gain.

It does not touch ANY canonical artifact: all outputs go to
data/processed/bge_revalidation/. It deliberately does not call the package
pipeline modules (generate_h2_profiles, etc.), because those write to fixed
paths shared with the canonical GTE artifacts.

Inputs:  data/raw/{books,albums}_filtered.jsonl,
         data/processed/{movies,games}/*_texts.json, user_book_history.json,
         canonical_test_users*.json
Outputs: data/processed/bge_revalidation/ — *_embeddings_bge.npy,
         *_catalog_whitened_bge.npy, whitening_*_bge.npz, diagnostics_bge.json,
         h2_<domain>_bge.npy (+ index)

Usage:
    python -m experiments.encoder_sweep.bge_revalidation
"""

import json
import time
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from acw.settings import ROOT  # noqa: E402
from acw.whitening import ZCAWhitening

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
OUT = PROC / "bge_revalidation"
OUT.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "BAAI/bge-large-en-v1.5"
SEED = 42
MAX_BOOKS_PROFILE = 10


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load_jsonl_texts(path: Path, id_field: str):
    ids, texts = [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            ids.append(d[id_field])
            texts.append(d["text"])
    return ids, texts


def embed_catalog(model, name, ids, texts):
    emb_path = OUT / f"{name}_embeddings_bge.npy"
    idx_path = OUT / f"{name}_id_to_idx_bge.json"
    if emb_path.exists() and idx_path.exists():
        log(f"{name}: embeddings already exist, skipping.")
        return np.load(emb_path), json.load(open(idx_path))

    log(f"{name}: embedding {len(texts):,} items with {MODEL_NAME}...")
    t0 = time.time()
    emb = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)
    log(f"{name}: done in {time.time() - t0:.1f}s, shape={emb.shape}")

    id_to_idx = {iid: i for i, iid in enumerate(ids)}
    np.save(emb_path, emb)
    json.dump(id_to_idx, open(idx_path, "w"))
    return emb, id_to_idx


def mean_cos_sim(X, sample=5000, seed=SEED):
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    S = X[idx]
    sim = S @ S.T
    iu = np.triu_indices_from(sim, k=1)
    return float(sim[iu].mean())


def eigen_diag(X):
    Xc = X.astype(np.float64) - X.mean(axis=0)
    cov = np.cov(Xc.T)
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = eigvals[::-1]
    return {
        "eig_min": float(eigvals.min()),
        "eig_max": float(eigvals.max()),
        "effective_rank_gt_0.01": int((eigvals > 0.01).sum()),
        "condition_number": float(eigvals.max() / max(eigvals.min(), 1e-300)),
    }


def build_h2_profiles(book_emb, book_id_to_idx, history, test_users):
    profiles, valid_users = [], []
    for uid in test_users:
        books = history.get(uid, [])[-MAX_BOOKS_PROFILE:]
        indices = [book_id_to_idx[b] for b in books if b in book_id_to_idx]
        if not indices:
            continue
        v = book_emb[indices].mean(axis=0)
        v = v / (np.linalg.norm(v) + 1e-12)
        profiles.append(v.astype(np.float32))
        valid_users.append(uid)
    return np.array(profiles, dtype=np.float32), valid_users


def main():
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    # ── 1. Catalogs ──────────────────────────────────────────────────────
    book_ids, book_texts = load_jsonl_texts(RAW / "books_filtered.jsonl", "parent_asin")
    album_ids, album_texts = load_jsonl_texts(RAW / "albums_filtered.jsonl", "parent_asin")
    movies_texts_d = json.load(open(PROC / "movies" / "movies_texts.json"))
    games_texts_d = json.load(open(PROC / "games" / "games_texts.json"))

    book_emb, book_id_to_idx = embed_catalog(model, "books", book_ids, book_texts)
    album_emb, album_id_to_idx = embed_catalog(model, "albums", album_ids, album_texts)
    movies_emb, movies_id_to_idx = embed_catalog(
        model, "movies", list(movies_texts_d.keys()), list(movies_texts_d.values())
    )
    games_emb, games_id_to_idx = embed_catalog(
        model, "games", list(games_texts_d.keys()), list(games_texts_d.values())
    )

    catalogs = {
        "music": (album_emb, album_id_to_idx),
        "movies": (movies_emb, movies_id_to_idx),
        "games": (games_emb, games_id_to_idx),
    }

    # ── 2. Whitening + diagnostics per domain ───────────────────────────
    diagnostics = {}
    for domain, (emb, _) in catalogs.items():
        raw_sim = mean_cos_sim(emb)
        zca = ZCAWhitening().fit(emb.astype(np.float64))
        whitened = zca.transform(emb)
        white_sim = mean_cos_sim(whitened)
        eig = eigen_diag(emb)
        diagnostics[domain] = {
            "n_items": emb.shape[0],
            "raw_mean_cos_sim": raw_sim,
            "whitened_mean_cos_sim": white_sim,
            **eig,
        }
        np.save(OUT / f"{domain}_catalog_whitened_bge.npy", whitened)
        zca.save(OUT / f"whitening_{domain}_bge.npz")
        log(
            f"{domain}: raw_sim={raw_sim:.4f} whitened_sim={white_sim:.4f} "
            f"eff_rank={eig['effective_rank_gt_0.01']} cond={eig['condition_number']:.3e}"
        )

    json.dump(diagnostics, open(OUT / "diagnostics_bge.json", "w"), indent=2)

    # ── 3. H2 profiles per domain (raw, music-source history) ───────────
    history = json.load(open(RAW / "user_book_history.json"))
    canonical = {
        "music": json.load(open(PROC / "canonical_test_users.json")),
        "movies": json.load(open(PROC / "canonical_test_users_movies.json")),
        "games": json.load(open(PROC / "canonical_test_users_games.json")),
    }

    for domain, test_users in canonical.items():
        h2, valid_users = build_h2_profiles(book_emb, book_id_to_idx, history, test_users)
        np.save(OUT / f"h2_{domain}_bge.npy", h2)
        idx = {u: i for i, u in enumerate(valid_users)}
        json.dump(idx, open(OUT / f"h2_{domain}_bge_index.json", "w"))
        log(f"H2 {domain}: {len(valid_users):,}/{len(test_users):,} valid users, shape={h2.shape}")

    log("Done. Outputs in data/processed/bge_revalidation/")


if __name__ == "__main__":
    main()
