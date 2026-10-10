"""
experiments/robustness/covariance_controls.py
=============================================
Covariance controls for asymmetric catalog whitening (music domain, H2
profiles, canonical validation users). Isolates three counter-arguments
(the JSON keys of each block are given in parentheses):

  Catalog subsample ("C1_subsample_curve"): does the catalog-side whitening
       gain come from the geometry (catalog vs. profile side) or simply from
       having more samples (n) to estimate the covariance? Fits ZCA on
       random subsamples of the album catalog at the profile-side sizes
       (1,792 / 2,731) and at intermediate sizes, applies it to the full
       catalog and re-evaluates H2.

  Book-fitted ("neutral") whitening ("H2_raw_catalog_whitened_neutral_books*"):
       applies to the album catalog the whitening fitted on the BOOK catalog
       (n=199,712), and on a book subsample size-matched to the album
       catalog, instead of the whitening fitted on the albums themselves.
       Separates "which side is whitened" from "which data the transform is
       fitted on".

  Covariance conditioning ("M4_eigen"): condition number and eigenvalue
       spectrum of Σ_profile (H2 raw, n=2,731) vs. Σ_items (albums,
       n=67,986) vs. Σ_items subsampled to n=2,731 (same n as the profiles);
       supports or refutes the "semantic collapse" reading against plain
       estimation noise.

Supports: the sample-size, catalog-swap and covariance-conditioning
controls of the catalog-whitening gain.

Reuses the standard evaluation harness (src/acw/evaluation) and ZCAWhitening
(src/acw/whitening.py) without modifying any canonical artifact. Nothing is
written to data/processed; all intermediate arrays stay in memory.

Inputs: data/processed/{album,book}_embeddings.npy, album_id_to_idx.json,
        user_profiles_h2.npy (+ index).
Output: results/covariance_controls.json

Usage:
    python -m experiments.robustness.covariance_controls
"""

from __future__ import annotations

import json
import time

import numpy as np

from acw.evaluation.config import resolve_paths  # noqa: E402
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users  # noqa: E402
from acw.evaluation.metrics import ndcg_at_k, precision_at_k  # noqa: E402
from acw.settings import ROOT  # noqa: E402
from acw.whitening import ZCAWhitening  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ALIGNMENT = PROCESSED / "alignment"
SEED = 42
K_VALUES = [5, 10, 20]


def evaluate(profiles, idx_map, items_emb, idx_to_item, genre_gt, test_users, label):
    max_k = max(K_VALUES)
    metrics = {f"nDCG@{k}": [] for k in K_VALUES}
    metrics.update({f"P@{k}": [] for k in K_VALUES})
    n_eval = 0
    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        row = idx_map.get(uid)
        if row is None:
            continue
        sims = items_emb @ profiles[row]
        top = np.argsort(sims)[::-1][:max_k]
        ranked = [idx_to_item[i] for i in top]
        for k in K_VALUES:
            metrics[f"nDCG@{k}"].append(ndcg_at_k(ranked, relevant, k))
            metrics[f"P@{k}"].append(precision_at_k(ranked, relevant, k))
        n_eval += 1
    avgs = {m: float(np.mean(v)) for m, v in metrics.items()}
    avgs["n_eval"] = n_eval
    print(f"  {label:<55}: nDCG@10={avgs['nDCG@10']:.4f}  P@10={avgs['P@10']:.4f}  (n={n_eval})")
    return avgs


def eigen_report(cov: np.ndarray, label: str) -> dict:
    eigvals = np.linalg.eigvalsh(cov)
    eigvals = np.sort(eigvals)[::-1]
    cond = float(eigvals[0] / max(eigvals[-1], 1e-12))
    report = {
        "label": label,
        "n_dims": int(cov.shape[0]),
        "lambda_max": float(eigvals[0]),
        "lambda_min": float(eigvals[-1]),
        "condition_number": cond,
        "effective_rank_gt_0.01": int((eigvals > 0.01).sum()),
        "frac_variance_top10": float(eigvals[:10].sum() / eigvals.sum()),
    }
    print(
        f"  {label:<40}: cond={cond:.3e}  lambda_max={eigvals[0]:.4f}  "
        f"lambda_min={eigvals[-1]:.6e}  eff_rank(>0.01)={report['effective_rank_gt_0.01']}  "
        f"top10_var_frac={report['frac_variance_top10']:.3f}"
    )
    return report


def main():
    t0 = time.time()
    cfg = resolve_paths(domain="music")
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)
    print(f"Val users (music, canonical validation set): {len(test_users)}")

    album_emb_path = PROCESSED / "album_embeddings.npy"
    album_id_to_idx = json.load(open(PROCESSED / "album_id_to_idx.json"))
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}

    album_emb = np.load(album_emb_path).astype(np.float32)
    book_emb = np.load(PROCESSED / "book_embeddings.npy").astype(np.float32)

    h2_profiles = np.load(PROCESSED / "user_profiles_h2.npy").astype(np.float32)
    h2_idx = json.load(open(PROCESSED / "user_profile_index_h2.json"))

    results: dict[str, dict] = {}

    # ── Reference baselines (recomputed here to guarantee exact parity) ──
    print("\n[REFERENCE BASELINES]")
    results["H2_raw_catalog_raw"] = evaluate(
        h2_profiles,
        h2_idx,
        album_emb,
        idx_to_album,
        genre_gt,
        test_users,
        "H2 raw / RAW catalog (baseline)",
    )

    zca_full = ZCAWhitening()
    zca_full.fit(album_emb)
    album_emb_w_full = zca_full.transform(album_emb)
    results["H2_raw_catalog_whitened_full"] = evaluate(
        h2_profiles,
        h2_idx,
        album_emb_w_full,
        idx_to_album,
        genre_gt,
        test_users,
        "H2 raw / WHITENED catalog (full fit, n=67,986)",
    )

    # ── C1: catalog subsample to estimate Σ_items at profile-matched n ──
    print("\n[C1 — CATALOG SUBSAMPLE FOR Σ_items AT PROFILE-MATCHED n]")
    rng = np.random.default_rng(SEED)
    subsample_ns = [1792, 2731, 5000, 10000, 20000, 67986]
    c1_results = {}
    for n in subsample_ns:
        if n >= len(album_emb):
            sub = album_emb
        else:
            sub_idx = rng.choice(len(album_emb), size=n, replace=False)
            sub = album_emb[sub_idx]
        zca_n = ZCAWhitening()
        zca_n.fit(sub)
        album_emb_w_n = zca_n.transform(album_emb)
        label = f"H2 raw / WHITENED catalog (Σ fitted on n={n})"
        res = evaluate(
            h2_profiles, h2_idx, album_emb_w_n, idx_to_album, genre_gt, test_users, label
        )
        c1_results[str(n)] = res
    results["C1_subsample_curve"] = c1_results

    # ── Book-fitted ("neutral") whitening: ZCA fitted on BOOKS applied to the album catalog ──
    print("\n[NEUTRAL WHITENING (Σ_books applied to ALBUMS)]")
    zca_book = ZCAWhitening()
    zca_book.fit(book_emb)
    album_emb_w_neutral = zca_book.transform(album_emb)
    results["H2_raw_catalog_whitened_neutral_books"] = evaluate(
        h2_profiles,
        h2_idx,
        album_emb_w_neutral,
        idx_to_album,
        genre_gt,
        test_users,
        "H2 raw / WHITENED catalog (Σ_books, n=199,712, 'neutral')",
    )
    # Additional control: books subsampled to the size of the full album catalog,
    # to separate out-of-domain data from simply "more data".
    book_sub_idx = rng.choice(len(book_emb), size=67986, replace=False)
    zca_book_matched = ZCAWhitening()
    zca_book_matched.fit(book_emb[book_sub_idx])
    album_emb_w_neutral_matched = zca_book_matched.transform(album_emb)
    results["H2_raw_catalog_whitened_neutral_books_matched_n"] = evaluate(
        h2_profiles,
        h2_idx,
        album_emb_w_neutral_matched,
        idx_to_album,
        genre_gt,
        test_users,
        "H2 raw / WHITENED catalog (Σ_books subsample n=67,986, size-matched)",
    )

    # ── M4: conditioning of Σ_profile vs Σ_items (full and subsampled) ──
    print("\n[M4 — CONDITIONING OF Σ (eigenvalues, condition number)]")
    h2_eval_profiles = np.array(
        [h2_profiles[h2_idx[u]] for u in test_users if u in h2_idx], dtype=np.float64
    )
    cov_profile = np.cov((h2_eval_profiles - h2_eval_profiles.mean(0)).T)
    eig_results = {}
    eig_results["sigma_profile_n2731"] = eigen_report(cov_profile, "Σ_profile (H2 raw, n=2,731)")

    cov_items_full = np.cov((album_emb.astype(np.float64) - album_emb.astype(np.float64).mean(0)).T)
    eig_results["sigma_items_full_n67986"] = eigen_report(
        cov_items_full, "Σ_items (albums, n=67,986)"
    )

    sub_idx_2731 = rng.choice(len(album_emb), size=2731, replace=False)
    sub_2731 = album_emb[sub_idx_2731].astype(np.float64)
    cov_items_2731 = np.cov((sub_2731 - sub_2731.mean(0)).T)
    eig_results["sigma_items_matched_n2731"] = eigen_report(
        cov_items_2731, "Σ_items (albums, subsample n=2,731)"
    )

    results["M4_eigen"] = eig_results

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")

    out_path = ROOT / "results" / "covariance_controls.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()
