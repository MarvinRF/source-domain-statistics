"""
experiments/robustness/offdomain_covariance.py
==============================================
Non-cultural catalog control for the "encoder-level, not catalog-specific"
claim. The book-fitted whitening control
(experiments/robustness/covariance_controls.py) compares only TWO "creative"
catalogs (books, n=199,712; albums, n=67,986), both organised by
genre/subgenre/mood, so it cannot rule out the alternative hypothesis that
the shared geometry reflects the structure of cultural catalogs (two
catalogs of creative works with rich metadata) rather than a universal
property of the encoder.

This script extends that control with the covariance of a structurally
different catalog: Grocery_and_Gourmet_Food (Amazon Reviews 2023), n=67,986
(the SAME size as the album catalog, which also removes the sample-size
confound without further subsampling), with no genre/subgenre/mood taxonomy
(purely functional/ingredient categories; see
experiments/robustness/offdomain_catalog_build.py). It evaluates H2 on the
album catalog whitened with Σ_grocery, compares the conditioning of
Σ_grocery and Σ_albums, and runs paired per-user significance tests against
own-catalog whitening and against the raw catalog.

Supports: the off-domain (non-cultural catalog) control of the
encoder-level geometry claim.

Reuses evaluate()/eigen_report() from
experiments/robustness/covariance_controls.py and ZCAWhitening
(src/acw/whitening.py). Nothing is written to data/processed.

Inputs: data/processed/grocery/grocery_embeddings_qwen.npy (embedded from
        the output of offdomain_catalog_build.py),
        data/processed/{album,book}_embeddings.npy, user_profiles_h2.npy.
Output: results/offdomain_covariance.json

Usage:
    python -m experiments.robustness.offdomain_covariance
"""

from __future__ import annotations

import json
import time

import numpy as np

from acw.evaluation.config import resolve_paths  # noqa: E402
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users  # noqa: E402
from acw.evaluation.metrics import ndcg_at_k, precision_at_k  # noqa: E402
from acw.evaluation.significance_tests import run_pair  # noqa: E402
from acw.settings import ROOT  # noqa: E402
from acw.whitening import ZCAWhitening  # noqa: E402
from experiments.robustness.covariance_controls import eigen_report, evaluate  # noqa: E402


def per_user_scores(profiles, idx_map, items_emb, idx_to_item, genre_gt, users):
    n5, n10, p10 = {}, {}, {}
    for uid in users:
        relevant = genre_gt.get(uid)
        row = idx_map.get(uid)
        if not relevant or row is None:
            continue
        sims = items_emb @ profiles[row]
        top = np.argsort(sims)[::-1][:10]
        ranked = [idx_to_item[i] for i in top]
        n5[uid] = ndcg_at_k(ranked, relevant, 5)
        n10[uid] = ndcg_at_k(ranked, relevant, 10)
        p10[uid] = precision_at_k(ranked, relevant, 10)
    return n5, n10, p10


def aligned(*score_dicts):
    common = set(score_dicts[0])
    for d in score_dicts[1:]:
        common &= set(d)
    users = sorted(common)
    return users, [np.array([d[u] for u in users]) for d in score_dicts]


PROCESSED = ROOT / "data" / "processed"
GROCERY_DIR = PROCESSED / "grocery"
SEED = 42


def main():
    t0 = time.time()
    cfg = resolve_paths(domain="music")
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)
    print(f"Val users (music, canonical validation set): {len(test_users)}")

    album_id_to_idx = json.load(open(PROCESSED / "album_id_to_idx.json"))
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}
    album_emb = np.load(PROCESSED / "album_embeddings.npy").astype(np.float32)
    book_emb = np.load(PROCESSED / "book_embeddings.npy").astype(np.float32)

    grocery_emb_path = GROCERY_DIR / "grocery_embeddings_qwen.npy"
    grocery_emb = np.load(grocery_emb_path).astype(np.float32)
    print(f"Grocery catalog: {grocery_emb.shape}")
    assert grocery_emb.shape[0] == album_emb.shape[0], (
        "Expected grocery catalog to be size-matched to albums "
        f"(got {grocery_emb.shape[0]} vs {album_emb.shape[0]})"
    )

    h2_profiles = np.load(PROCESSED / "user_profiles_h2.npy").astype(np.float32)
    h2_idx = json.load(open(PROCESSED / "user_profile_index_h2.json"))

    results: dict[str, dict] = {}

    # ── Reference baselines (same numbers as the book-fitted control, recomputed for parity) ──
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

    zca_own = ZCAWhitening()
    zca_own.fit(album_emb)
    album_w_own = zca_own.transform(album_emb)
    results["H2_raw_catalog_whitened_own"] = evaluate(
        h2_profiles,
        h2_idx,
        album_w_own,
        idx_to_album,
        genre_gt,
        test_users,
        "H2 raw / WHITENED catalog (albums' own Σ, n=67,986)",
    )

    zca_book = ZCAWhitening()
    zca_book.fit(book_emb)
    album_w_book = zca_book.transform(album_emb)
    results["H2_raw_catalog_whitened_books"] = evaluate(
        h2_profiles,
        h2_idx,
        album_w_book,
        idx_to_album,
        genre_gt,
        test_users,
        "H2 raw / WHITENED catalog (Σ_books, n=199,712, creative catalog)",
    )

    # ── Central test: Σ_grocery (functional catalog, no genre/mood) ───────────
    print("\n[CENTRAL TEST — Σ_grocery APPLIED TO ALBUMS]")
    zca_grocery = ZCAWhitening()
    zca_grocery.fit(grocery_emb)
    album_w_grocery = zca_grocery.transform(album_emb)
    results["H2_raw_catalog_whitened_grocery"] = evaluate(
        h2_profiles,
        h2_idx,
        album_w_grocery,
        idx_to_album,
        genre_gt,
        test_users,
        "H2 raw / WHITENED catalog (Σ_grocery, n=67,986, NON-creative catalog)",
    )

    # ── Conditioning of Σ_grocery vs Σ_albums (same format as covariance_controls) ──
    print("\n[CONDITIONING OF Σ — albums vs grocery]")
    eig_results = {}
    cov_album = np.cov((album_emb.astype(np.float64) - album_emb.astype(np.float64).mean(0)).T)
    eig_results["sigma_album_n67986"] = eigen_report(cov_album, "Σ_albums (n=67,986)")
    cov_grocery = np.cov(
        (grocery_emb.astype(np.float64) - grocery_emb.astype(np.float64).mean(0)).T
    )
    eig_results["sigma_grocery_n67986"] = eigen_report(cov_grocery, "Σ_grocery (n=67.986)")
    results["eigen_conditioning"] = eig_results

    # ── Significance: Σ_grocery vs own Σ and vs raw (per-user, paired) ──
    print("\n[SIGNIFICANCE — Σ_grocery vs baselines, paired per user]")
    n5_g, n10_g, p10_g = per_user_scores(
        h2_profiles, h2_idx, album_w_grocery, idx_to_album, genre_gt, test_users
    )
    n5_o, n10_o, p10_o = per_user_scores(
        h2_profiles, h2_idx, album_w_own, idx_to_album, genre_gt, test_users
    )
    n5_r, n10_r, p10_r = per_user_scores(
        h2_profiles, h2_idx, album_emb, idx_to_album, genre_gt, test_users
    )

    users_go, (a5, a10, ap10, b5, b10, bp10) = aligned(n5_g, n10_g, p10_g, n5_o, n10_o, p10_o)
    res_vs_own = run_pair(
        pair_id="grocery_vs_own_whitening",
        family="noncreative_catalog_covariance",
        method_a="H2+ZCA(grocery)",
        method_b="H2+ZCA(own albums)",
        desc=f"Grocery-covariance vs own-covariance whitening, n={len(users_go)}",
        ndcg5_a=a5,
        ndcg10_a=a10,
        p10_a=ap10,
        ndcg5_b=b5,
        ndcg10_b=b10,
        p10_b=bp10,
    )
    users_gr, (c5, c10, cp10, d5, d10, dp10) = aligned(n5_g, n10_g, p10_g, n5_r, n10_r, p10_r)
    res_vs_raw = run_pair(
        pair_id="grocery_vs_raw",
        family="noncreative_catalog_covariance",
        method_a="H2+ZCA(grocery)",
        method_b="H2 raw",
        desc=f"Grocery-covariance whitening vs raw catalog, n={len(users_gr)}",
        ndcg5_a=c5,
        ndcg10_a=c10,
        p10_a=cp10,
        ndcg5_b=d5,
        ndcg10_b=d10,
        p10_b=dp10,
    )
    print(
        f"  vs own Σ    (0.3476): delta={res_vs_own['delta_ndcg10']:+.4f} "
        f"p_perm={res_vs_own['p_perm_raw']:.4g} r={res_vs_own['effect_size_r']:.3f} "
        f"({res_vs_own['effect_size_label']})"
    )
    print(
        f"  vs raw       (0.2691): delta={res_vs_raw['delta_ndcg10']:+.4f} "
        f"p_perm={res_vs_raw['p_perm_raw']:.4g} r={res_vs_raw['effect_size_r']:.3f} "
        f"({res_vs_raw['effect_size_label']})"
    )
    results["significance"] = {"vs_own_whitening": res_vs_own, "vs_raw": res_vs_raw}

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed / 60:.1f} min.")

    out_path = ROOT / "results" / "offdomain_covariance.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {out_path}")

    # ── Comparative summary ─────────────────────────────────────────────────
    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(
        f"  raw (no whitening)                     : {results['H2_raw_catalog_raw']['nDCG@10']:.4f}"
    )
    print(
        f"  whitened (own Σ, albums)               : {results['H2_raw_catalog_whitened_own']['nDCG@10']:.4f}"
    )
    print(
        f"  whitened (Σ_books, creative, n=199,712): {results['H2_raw_catalog_whitened_books']['nDCG@10']:.4f}"
    )
    print(
        f"  whitened (Σ_grocery, NON-creative, n=67,986): {results['H2_raw_catalog_whitened_grocery']['nDCG@10']:.4f}"
    )


if __name__ == "__main__":
    main()
