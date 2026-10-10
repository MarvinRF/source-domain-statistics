"""
experiments/encoder_sweep/screening.py
======================================
Inexpensive screening of effective rank / anisotropy for candidate embedding
backbones (E5-large-v2, all-MiniLM-L6-v2, nomic-embed-text-v1.5), used to
decide which ones are worth a full catalog embedding.

Uses a random sample of N_SAMPLE items from the album catalog (not the full
catalog), which is enough to estimate the effective rank and cheap to run.

Supports: the encoder sweep relating embedding anisotropy to the
catalog-whitening gain (backbone selection step).

Input:  data/raw/albums_filtered.jsonl
Output: data/processed/backbone_screening/screening_results.json and
        sample_emb_<model>.npy (nothing is written outside that directory).

Usage:
    python -m experiments.encoder_sweep.screening
"""

import json

import numpy as np
from sentence_transformers import SentenceTransformer

from acw.settings import ROOT  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "backbone_screening"
OUT.mkdir(parents=True, exist_ok=True)

N_SAMPLE = 8000
SEED = 42


def load_sample_texts(path, n=N_SAMPLE, seed=SEED):
    texts = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            texts.append(d["text"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(texts), size=min(n, len(texts)), replace=False)
    return [texts[i] for i in idx]


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
    eigvals = np.linalg.eigvalsh(cov)[::-1]
    return {
        "dim": X.shape[1],
        "eig_max": float(eigvals.max()),
        "eig_min": float(eigvals.min()),
        "effective_rank_gt_0.01": int((eigvals > 0.01).sum()),
        "effective_rank_frac": float((eigvals > 0.01).sum() / X.shape[1]),
        "condition_number": float(eigvals.max() / max(eigvals.min(), 1e-300)),
    }


def screen(name, prefix=""):
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    texts = load_sample_texts(RAW / "albums_filtered.jsonl")
    if prefix:
        texts = [prefix + t for t in texts]

    device = "cuda"
    m = SentenceTransformer(name, trust_remote_code=True, device=device)
    emb = m.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    diag = eigen_diag(emb)
    raw_sim = mean_cos_sim(emb)
    diag["raw_mean_cos_sim"] = raw_sim
    diag["n_sample"] = emb.shape[0]

    print(
        f"dim={diag['dim']}  eff_rank={diag['effective_rank_gt_0.01']} "
        f"({diag['effective_rank_frac']:.1%})  raw_sim={raw_sim:.4f}  "
        f"cond={diag['condition_number']:.2e}"
    )

    safe_name = name.replace("/", "_")
    np.save(OUT / f"sample_emb_{safe_name}.npy", emb)
    return diag


def main():
    results = {}
    results["e5-large-v2"] = screen("intfloat/e5-large-v2", prefix="passage: ")
    results["all-MiniLM-L6-v2"] = screen("sentence-transformers/all-MiniLM-L6-v2")
    results["nomic-embed-text-v1.5"] = screen(
        "nomic-ai/nomic-embed-text-v1.5", prefix="search_document: "
    )

    json.dump(results, open(OUT / "screening_results.json", "w"), indent=2)
    print("\n\nComparative summary:")
    print(f"{'Backbone':<25} {'dim':>6} {'eff_rank':>10} {'frac':>8} {'raw_sim':>10}")
    for name, d in results.items():
        print(
            f"{name:<25} {d['dim']:>6} {d['effective_rank_gt_0.01']:>10} "
            f"{d['effective_rank_frac']:>8.1%} {d['raw_mean_cos_sim']:>10.4f}"
        )


if __name__ == "__main__":
    main()
