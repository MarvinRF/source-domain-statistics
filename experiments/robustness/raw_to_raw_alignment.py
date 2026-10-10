"""
experiments/robustness/raw_to_raw_alignment.py
==============================================
"Raw-to-raw" comparison between catalog whitening and learned alignment.

Motivation: the learned alignment configurations are supervised against the
centroid of ALREADY WHITENED album embeddings and evaluated against the
ALREADY WHITENED catalog, so "no learned method beats whitening" is nearly
guaranteed by the objective itself (the supervision target IS the output of
whitening), rather than being a symmetric "learning vs. whitening from the
same starting point" comparison. This is structural: most of the learned
configurations are supervised against a whitened target, not merely
evaluated on a whitened catalog.

This script trains PW and MoE-PW (the most informative pair: the simplest
learned method and the best of the family) with both the supervision target
AND the evaluation pointing to the RAW (unwhitened) album catalog, via the
--items-emb-path override supported by parametric_whitening.py and
moe_whitening.py. Artifacts are saved with a `_rawtarget` suffix and never
overwrite the canonical files.

Results are compared against H2-raw (plain mean pooling) and H2-whitened
(the catalog-whitening baseline, 0.3476), both recomputed on the SAME
ordered population so that the paired test is positionally aligned
(run_pair expects paired arrays in the same user order). p-values are
Holm-corrected across all pairs.

Supports: the raw-target control for the learned-alignment baselines (does
learned alignment beat whitening when it is not supervised by whitened
targets?).

Inputs: data/processed/album_embeddings.npy,
        data/processed/alignment/album_embeddings_whitened.npy,
        user_profiles_{h2,h3,h4}.npy (+ indices), canonical_val_users.json.
Output: results/statistical_tests/raw_to_raw_alignment_test.json (plus the
        `_rawtarget` profile arrays in data/processed/alignment/).

Usage:
    python -m experiments.robustness.raw_to_raw_alignment
"""

import json

import numpy as np
from statsmodels.stats.multitest import multipletests

from acw.baselines import moe_whitening as moe_mod
from acw.baselines import parametric_whitening as pw_mod
from acw.evaluation.significance_tests import run_pair
from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ALIGNMENT_DIR = PROCESSED / "alignment"
RAW_ALBUM_PATH = PROCESSED / "album_embeddings.npy"
RESULTS_DIR = ROOT / "results" / "statistical_tests"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def per_user_scores(profiles, idx_map, items_emb, idx_to_item, genre_gt, users):
    """Per-user nDCG@5, nDCG@10, P@10, keyed by user_id (mirrors evaluate()
    in parametric_whitening.py, but keeps individual scores instead of only
    the mean -- needed for the paired significance test)."""
    n5, n10, p10 = {}, {}, {}
    for uid in users:
        relevant = genre_gt.get(uid)
        row = idx_map.get(uid)
        if not relevant or row is None:
            continue
        sims = items_emb @ profiles[row]
        top = np.argsort(sims)[::-1][:10]
        ranked = [idx_to_item[j] for j in top if j in idx_to_item]
        n5[uid] = pw_mod.ndcg_at_k(ranked, relevant, 5)
        n10[uid] = pw_mod.ndcg_at_k(ranked, relevant, 10)
        p10[uid] = pw_mod.precision_at_k(ranked, relevant, 10)
    return n5, n10, p10


def aligned(*score_dicts):
    """Intersect user_ids across all dicts; return positionally-aligned arrays
    (run_pair requires paired/same-order samples)."""
    common = set(score_dicts[0])
    for d in score_dicts[1:]:
        common &= set(d)
    users = sorted(common)
    return users, [np.array([d[u] for u in users]) for d in score_dicts]


def main():
    album_raw = np.load(RAW_ALBUM_PATH)
    album_whitened = np.load(ALIGNMENT_DIR / "album_embeddings_whitened.npy")
    with open(PROCESSED / "album_id_to_idx.json") as f:
        album_id_to_idx = json.load(f)
    idx_to_album = {v: k for k, v in album_id_to_idx.items()}

    genre_gt = pw_mod.build_genre_ground_truth()

    with open(PROCESSED / "canonical_val_users.json") as f:
        val_users = json.load(f)

    with open(PROCESSED / "user_profile_index_h2.json") as f:
        prof_idx_h2 = json.load(f)
    h2_profiles = np.load(PROCESSED / "user_profiles_h2.npy")

    h2_raw_n5, h2_raw_n10, h2_raw_p10 = per_user_scores(
        h2_profiles, prof_idx_h2, album_raw, idx_to_album, genre_gt, val_users
    )
    h2_w_n5, h2_w_n10, h2_w_p10 = per_user_scores(
        h2_profiles, prof_idx_h2, album_whitened, idx_to_album, genre_gt, val_users
    )
    print(
        f"H2 raw:      mean nDCG@10 = {np.mean(list(h2_raw_n10.values())):.4f}  (n={len(h2_raw_n10)})"
    )
    print(
        f"H2 whitened: mean nDCG@10 = {np.mean(list(h2_w_n10.values())):.4f}  (n={len(h2_w_n10)})"
    )

    all_results = []

    for heuristic in ["h3", "h4"]:
        print(f"\n{'=' * 70}\n{heuristic.upper()} -- raw-to-raw training\n{'=' * 70}")

        print(f"\n[{heuristic}] Training PW (raw target)...")
        pw_mod.train(
            heuristic=heuristic,
            domain="music",
            device_str="cpu",
            items_emb_path=RAW_ALBUM_PATH,
            epochs=200,
        )
        print(f"\n[{heuristic}] Training MoE-PW (raw target)...")
        moe_mod.train(
            heuristic=heuristic,
            domain="music",
            device_str="cpu",
            items_emb_path=RAW_ALBUM_PATH,
            epochs=200,
        )

        with open(PROCESSED / f"user_profile_index_{heuristic}.json") as f:
            prof_idx_h = json.load(f)
        users_h = [u for u in val_users if u in prof_idx_h]

        pw_profiles = np.load(ALIGNMENT_DIR / f"user_profiles_{heuristic}_pw_rawtarget.npy")
        moe_profiles = np.load(ALIGNMENT_DIR / f"user_profiles_{heuristic}_moe_rawtarget.npy")

        for label, profiles in [("PW", pw_profiles), ("MoE-PW", moe_profiles)]:
            n5, n10, p10 = per_user_scores(
                profiles, prof_idx_h, album_raw, idx_to_album, genre_gt, users_h
            )
            method_a = f"{heuristic.upper()}+{label}-rawtarget"
            print(f"\n{method_a}: mean nDCG@10 = {np.mean(list(n10.values())):.4f}  (n={len(n10)})")

            users_vs_raw, (a5, a10, ap10, b5, b10, bp10) = aligned(
                n5, n10, p10, h2_raw_n5, h2_raw_n10, h2_raw_p10
            )
            res_vs_raw = run_pair(
                pair_id=f"{heuristic}_{label}_rawtarget_vs_H2raw",
                family="raw_to_raw_alignment",
                method_a=method_a,
                method_b="H2-raw",
                desc=f"{label} raw-to-raw vs H2 raw mean-pool, n={len(users_vs_raw)}",
                ndcg5_a=a5,
                ndcg10_a=a10,
                p10_a=ap10,
                ndcg5_b=b5,
                ndcg10_b=b10,
                p10_b=bp10,
            )

            users_vs_w, (c5, c10, cp10, d5, d10, dp10) = aligned(
                n5, n10, p10, h2_w_n5, h2_w_n10, h2_w_p10
            )
            res_vs_whitened = run_pair(
                pair_id=f"{heuristic}_{label}_rawtarget_vs_H2whitened",
                family="raw_to_raw_alignment",
                method_a=method_a,
                method_b="H2-whitened",
                desc=f"{label} raw-to-raw vs H2 catalog-whitened, n={len(users_vs_w)}",
                ndcg5_a=c5,
                ndcg10_a=c10,
                p10_a=cp10,
                ndcg5_b=d5,
                ndcg10_b=d10,
                p10_b=dp10,
            )

            print(
                f"  vs H2-raw      ({res_vs_raw['mean_ndcg10_b']:.4f}): "
                f"delta={res_vs_raw['delta_ndcg10']:+.4f} p_perm={res_vs_raw['p_perm_raw']:.4g} "
                f"r={res_vs_raw['effect_size_r']:.3f} ({res_vs_raw['effect_size_label']})"
            )
            print(
                f"  vs H2-whitened ({res_vs_whitened['mean_ndcg10_b']:.4f}): "
                f"delta={res_vs_whitened['delta_ndcg10']:+.4f} p_perm={res_vs_whitened['p_perm_raw']:.4g} "
                f"r={res_vs_whitened['effect_size_r']:.3f} ({res_vs_whitened['effect_size_label']})"
            )

            all_results.append(
                {
                    "heuristic": heuristic,
                    "method": label,
                    "vs_H2_raw": res_vs_raw,
                    "vs_H2_whitened": res_vs_whitened,
                }
            )

    # ── Holm-Bonferroni correction across the whole family (k=8 pairs) ─────
    flat_pairs = []
    for entry in all_results:
        flat_pairs.append(entry["vs_H2_raw"])
        flat_pairs.append(entry["vs_H2_whitened"])
    pvals = [p["p_perm_raw"] for p in flat_pairs]
    reject, p_holm, _, _ = multipletests(pvals, alpha=0.05, method="holm")
    for pair, p_h, sig in zip(flat_pairs, p_holm, reject):
        pair["p_perm_holm"] = round(float(p_h), 6)
        pair["significant_perm_holm"] = bool(sig)

    print(f"\n{'=' * 70}\nHolm-corrected (k={len(flat_pairs)}) summary\n{'=' * 70}")
    for pair in flat_pairs:
        print(
            f"  {pair['method_a']} vs {pair['method_b']}: delta={pair['delta_ndcg10']:+.4f} "
            f"p_holm={pair['p_perm_holm']:.4g} significant={pair['significant_perm_holm']} "
            f"r={pair['effect_size_r']:.3f} ({pair['effect_size_label']})"
        )

    out_path = RESULTS_DIR / "raw_to_raw_alignment_test.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
