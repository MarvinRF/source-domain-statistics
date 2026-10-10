"""
experiments/robustness/pca_vs_zca_and_collapse.py
=================================================
Two controls for asymmetric catalog whitening (music domain, canonical
validation users, genre ground truth):

  PCA vs. ZCA (JSON key "M3"): is the choice of ZCA consequential or a
       convention? Compares catalog whitening by ZCA (W = V Λ^{-1/2} V^T)
       and by PCA (W = Λ^{-1/2} V^T, i.e. the same whitening rotated by V^T)
       under two scoring modes:
         - asymmetric (default): whitened catalog × raw profile
         - symmetric (literal Eq. 1): the profile is also passed through
           the catalog whitening transform
       In symmetric mode PCA ≡ ZCA by construction (an orthogonal V
       preserves cosine), which serves as a sanity check.

  Profile collapse (JSON key "M4"): mean pairwise cosine similarity among
       H2, H3 and H4 validation profiles (raw encoder space), plus the
       fraction of variance in the top-10 eigenvalues; the album catalog is
       included as a reference.

Supports: the ablation showing that the whitening gain does not depend on
the ZCA rotation in asymmetric scoring, and the anisotropy/collapse
characterisation of the user-profile families.

Inputs: data/processed/album_embeddings.npy (+ index) and
        user_profiles_{h2,h3,h4}.npy (+ indices). Nothing is written to
        data/processed.
Output: results/pca_vs_zca_and_collapse.json

Usage:
    python -m experiments.robustness.pca_vs_zca_and_collapse
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import wilcoxon

from acw.evaluation.config import resolve_paths  # noqa: E402
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users  # noqa: E402
from acw.evaluation.metrics import ndcg_at_k  # noqa: E402
from acw.evaluation.significance import rank_biserial  # noqa: E402
from acw.settings import ROOT  # noqa: E402
from acw.whitening import EPS, ZCAWhitening  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SEED = 42
FAMILIES = {
    "H2": ("user_profiles_h2.npy", "user_profile_index_h2.json"),
    "H3": ("user_profiles_h3.npy", "user_profile_index_h3.json"),
    "H4": ("user_profiles_h4.npy", "user_profile_index_h4.json"),
}


def l2(X):
    return X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), EPS)


def per_user_ndcg10(P, items, idx_to_item, relevant_sets, batch=512):
    """Per-user nDCG@10; P (n_users × D) is row-aligned with relevant_sets."""
    out = np.empty(len(P))
    for s in range(0, len(P), batch):
        S = P[s : s + batch] @ items.T
        top = np.argpartition(-S, 10, axis=1)[:, :10]
        for r in range(len(S)):
            t = top[r][np.argsort(-S[r, top[r]])]
            out[s + r] = ndcg_at_k([idx_to_item[i] for i in t], relevant_sets[s + r], 10)
    return out


def mean_pairwise_cos(X):
    X = l2(X.astype(np.float64))
    n = len(X)
    s = X.sum(0)
    return float((s @ s - n) / (n * (n - 1)))  # off-diagonal mean without building the n×n matrix


def top10_var(X):
    ev = np.linalg.eigvalsh(np.cov((X - X.mean(0)).T.astype(np.float64)))[::-1]
    return float(ev[:10].sum() / ev.sum())


def main():
    cfg = resolve_paths(domain="music")
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)

    album_id_to_idx = json.load(open(PROCESSED / "album_id_to_idx.json"))
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}
    album = np.load(PROCESSED / "album_embeddings.npy").astype(np.float32)

    zca = ZCAWhitening().fit(album)
    # PCA whitening = same scaling without the back-rotation: W_pca = V^T W_zca
    ev, V = np.linalg.eigh(np.cov((album - zca.mean_).T.astype(np.float64)))
    W_pca = np.diag(1.0 / np.sqrt(ev + EPS)) @ V.T
    assert np.allclose(V @ W_pca, zca.W_, atol=1e-6 * np.abs(zca.W_).max())

    def pca_transform(X):
        return l2((X.astype(np.float64) - zca.mean_) @ W_pca.T).astype(np.float32)

    items = {"raw": album, "zca": zca.transform(album), "pca": pca_transform(album)}

    results = {"M3": {}, "M4": {}}
    rng = np.random.default_rng(SEED)
    cat_sample = album[rng.choice(len(album), 5000, replace=False)]
    results["M4"]["catalog_sample5000"] = {
        "mean_pairwise_cos": mean_pairwise_cos(cat_sample),
        "top10_var": top10_var(album),
    }

    for fam, (pf, ixf) in FAMILIES.items():
        prof = np.load(PROCESSED / pf).astype(np.float32)
        idx = json.load(open(PROCESSED / ixf))
        users = [u for u in test_users if u in idx]
        P = prof[[idx[u] for u in users]]
        rel = [genre_gt[u] for u in users]

        scores = {
            "raw": per_user_ndcg10(P, items["raw"], idx_to_album, rel),
            "zca_asym": per_user_ndcg10(P, items["zca"], idx_to_album, rel),
            "pca_asym": per_user_ndcg10(P, items["pca"], idx_to_album, rel),
            "zca_sym": per_user_ndcg10(zca.transform(P), items["zca"], idx_to_album, rel),
            "pca_sym": per_user_ndcg10(pca_transform(P), items["pca"], idx_to_album, rel),
        }
        p = wilcoxon(scores["zca_asym"], scores["pca_asym"], zero_method="pratt").pvalue
        r = rank_biserial(scores["zca_asym"], scores["pca_asym"])
        results["M3"][fam] = {
            "n": len(users),
            **{k: float(v.mean()) for k, v in scores.items()},
            "zca_vs_pca_asym_p": float(p),
            "zca_vs_pca_asym_r": r,
        }
        results["M4"][fam] = {
            "n": len(users),
            "mean_pairwise_cos": mean_pairwise_cos(P),
            "top10_var": top10_var(P),
        }
        print(fam, json.dumps(results["M3"][fam], indent=1), json.dumps(results["M4"][fam]))

    print("catalog", results["M4"]["catalog_sample5000"])
    out = ROOT / "results" / "pca_vs_zca_and_collapse.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
