"""
experiments/robustness/popularity_control.py
============================================
Popularity control for asymmetric catalog whitening: is the
catalog-whitening gain an artefact of popularity bias? Uses the canonical
music protocol, H2 raw vs. H2 + catalog ZCA (asymmetric scoring, Eq. 1).
Item popularity = rating_number (Amazon Reviews 2023,
albums_filtered.jsonl).

  A — Popularity of what is recommended: mean popularity percentile of the
      top-10 (raw vs. ZCA), plus a baseline that ranks the catalog by
      popularity alone (same list for every user).

  B — Catalog stratified by popularity decile: the catalog is restricted to
      each decile (ZCA still fitted on the full catalog, as in the main
      setup), each user's relevant set is intersected with the decile, and
      raw vs. ZCA nDCG@10 is compared with a paired Wilcoxon-Pratt test,
      Holm-corrected over k=10. Deciles are by rank (ties broken by the
      stable catalog order).

  C — Users stratified by the popularity of their own consumption: mean
      popularity percentile of the albums in the user's exact ground truth,
      in quintiles; raw → ZCA gain per quintile, Holm-corrected over k=5.

Supports: the popularity-stratified control of the catalog-whitening gain.

Reuses per_user_ndcg10/rank_biserial from
experiments/robustness/pca_vs_zca_and_collapse.py and the standard
evaluation harness. Nothing is written to data/processed.

Inputs: data/processed/album_embeddings.npy (+ index),
        user_profiles_h2.npy (+ index), music item metadata and ground truth.
Output: results/popularity_control.json

Usage:
    python -m experiments.robustness.popularity_control
"""

from __future__ import annotations

import json

import numpy as np
from scipy.stats import rankdata, wilcoxon
from statsmodels.stats.multitest import multipletests

from acw.evaluation.config import resolve_paths  # noqa: E402
from acw.evaluation.loader import build_genre_ground_truth, sample_test_users  # noqa: E402
from acw.evaluation.metrics import ndcg_at_k  # noqa: E402
from acw.settings import ROOT  # noqa: E402
from acw.whitening import ZCAWhitening  # noqa: E402
from experiments.robustness.pca_vs_zca_and_collapse import (  # noqa: E402
    per_user_ndcg10,
    rank_biserial,
)

PROCESSED = ROOT / "data" / "processed"
N_DECILES = 10
N_USER_BINS = 5


def paired(raw, zca):
    p = wilcoxon(zca, raw, zero_method="pratt").pvalue
    return {
        "n": len(raw),
        "raw": float(raw.mean()),
        "zca": float(zca.mean()),
        "delta": float((zca - raw).mean()),
        "p": float(p),
        "r": rank_biserial(zca, raw),
    }


def holm(rows):
    adj = multipletests([r["p"] for r in rows], method="holm")[1]
    for r, a in zip(rows, adj):
        r["p_holm"] = float(a)
    return rows


def main():
    cfg = resolve_paths(domain="music")
    genre_gt = build_genre_ground_truth(cfg)
    test_users = sample_test_users(genre_gt, cfg)
    gt_exact = json.load(open(cfg.gt_path))

    album_id_to_idx = json.load(open(PROCESSED / "album_id_to_idx.json"))
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}
    album = np.load(PROCESSED / "album_embeddings.npy").astype(np.float32)
    album_w = ZCAWhitening().fit(album).transform(album)

    pop_by_id = {}
    with open(cfg.items_meta_path) as f:
        for line in f:
            it = json.loads(line)
            pop_by_id[it["parent_asin"]] = it.get("rating_number") or 0
    pop = np.array([pop_by_id.get(idx_to_album[i], 0) for i in range(len(album))], dtype=float)
    pct = rankdata(pop) / len(pop)  # per-item popularity percentile in (0, 1]
    pct_by_id = {idx_to_album[i]: pct[i] for i in range(len(album))}

    prof = np.load(PROCESSED / "user_profiles_h2.npy").astype(np.float32)
    idx = json.load(open(PROCESSED / "user_profile_index_h2.json"))
    users = [u for u in test_users if u in idx]
    P = prof[[idx[u] for u in users]]
    rel = [genre_gt[u] for u in users]
    results = {"n_users": len(users)}

    # ── A: top-10 popularity ──
    def top10_pct(items):
        S = P @ items.T
        top = np.argpartition(-S, 10, axis=1)[:, :10]
        return pct[top].mean(1)

    raw_pct, zca_pct = top10_pct(album), top10_pct(album_w)
    pop_top = [idx_to_album[i] for i in np.argsort(-pop, kind="stable")[:10]]
    pop_only = np.array([ndcg_at_k(pop_top, r, 10) for r in rel])
    results["A_top10_popularity"] = {
        "raw_mean_pct": float(raw_pct.mean()),
        "zca_mean_pct": float(zca_pct.mean()),
        "p": float(wilcoxon(zca_pct, raw_pct, zero_method="pratt").pvalue),
        "r": rank_biserial(zca_pct, raw_pct),
        "popularity_only_ndcg10": float(pop_only.mean()),
    }
    print("A", json.dumps(results["A_top10_popularity"], indent=1))

    # ── B: catalog restricted to each decile ──
    order = np.argsort(pop, kind="stable")
    rows = []
    for d, sub in enumerate(np.array_split(order, N_DECILES)):
        sub_ids = [idx_to_album[i] for i in sub]
        sub_set = set(sub_ids)
        keep = [j for j, r in enumerate(rel) if r & sub_set]
        rel_d = [rel[j] & sub_set for j in keep]
        raw = per_user_ndcg10(P[keep], album[sub], sub_ids, rel_d)
        zca = per_user_ndcg10(P[keep], album_w[sub], sub_ids, rel_d)
        rows.append(
            {
                "decile": d + 1,
                "rating_number_min": float(pop[sub].min()),
                "rating_number_max": float(pop[sub].max()),
                **paired(raw, zca),
            }
        )
    results["B_catalog_deciles"] = holm(rows)
    for r in rows:
        print("B", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})

    # ── C: users by popularity of their own consumption ──
    raw_all = per_user_ndcg10(P, album, idx_to_album, rel)
    zca_all = per_user_ndcg10(P, album_w, idx_to_album, rel)
    user_pop = np.array(
        [np.mean([pct_by_id[a] for a in gt_exact[u] if a in pct_by_id] or [np.nan]) for u in users]
    )
    ok = ~np.isnan(user_pop)
    bins = np.array_split(np.where(ok)[0][np.argsort(user_pop[ok], kind="stable")], N_USER_BINS)
    rows = [
        {
            "quintile": q + 1,
            "user_pop_pct_mean": float(user_pop[b].mean()),
            **paired(raw_all[b], zca_all[b]),
        }
        for q, b in enumerate(bins)
    ]
    results["C_user_quintiles"] = holm(rows)
    results["overall"] = paired(raw_all, zca_all)
    for r in rows:
        print("C", {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    print("overall", results["overall"])

    out = ROOT / "results" / "popularity_control.json"
    json.dump(results, open(out, "w"), indent=2)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
