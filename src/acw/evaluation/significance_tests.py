"""Paired significance tests between music-domain methods, read from the
per-user CSVs produced by ``acw.evaluation.per_user_metrics``.

Protocol:
  - Wilcoxon signed-rank (zero_method='pratt')
  - Vectorised permutation test (scipy.stats.permutation_test)
  - Effect size: matched-pairs rank-biserial correlation r
  - Holm-Bonferroni correction within each family of hypotheses

Families of hypotheses:
  Family 1 (k=6): comparisons on the per-user CSVs
  Family 2 (k=2): neural-variant comparisons recomputed on the val split
  Sanity checks: H2 vs Random, H5 vs Random (uncorrected)

References:
  Demšar (2006) JMLR — Wilcoxon + Holm
  Smucker et al. (2007) CIKM — permutation test
  Pratt (1959) JASA — zero_method='pratt'
  Kerby (2014) — rank-biserial r

Usage
-----
    python -m acw.evaluation.significance_tests
    python -m acw.evaluation.significance_tests --n-permutations 10000
"""

from __future__ import annotations

import argparse
import csv
import logging
import sys

import numpy as np
from scipy.stats import permutation_test, wilcoxon

from acw.evaluation.significance import effect_label, holm_rows, rank_biserial
from acw.settings import ROOT  # noqa: E402

ST_DIR = ROOT / "results" / "statistical_tests"
ST_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s.%(msecs)03d  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

SEED = 42


# ─── Hypothesis families and pairs (fixed before testing) ────────────────────

FAMILY_1 = [
    # (pair_id, method_a, method_b, description)
    ("P1", "H5-alpha095", "H2", "main contribution: H5 vs H2"),
    ("P2", "H5-alpha095", "H3-PW-ZB", "H5 vs best LLM-aligned"),
    ("P3", "H3-PW-Linear", "H3-PW-ZB", "PW-Linear vs PW-ZB"),
    ("P4", "H3-PW-ZB", "H3-sg-pw-zb", "specificity vs validity"),
    ("P5", "H3-PW-Linear", "H3-ZCA-profile", "PW gain over static ZCA"),
    ("P6", "H4-PW-Linear", "H3-PW-Linear", "H4 vs H3 after alignment"),
]

FAMILY_2 = [
    ("P7", "H3-NL-PW-ZB", "H3-PW-ZB-val", "best neural vs linear (val)"),
    ("P8", "H3-MoE-ZB", "H3-MoE-PW", "zca_book effect on MoE (val)"),
]

SANITY_CHECKS = [
    ("S1", "H2", "Random", "baseline > random"),
    ("S2", "H5-alpha095", "Random", "SOTA > random"),
]


# ─── Single-pair computation ─────────────────────────────────────────────────


def run_pair(
    pair_id: str,
    family: str,
    method_a: str,
    method_b: str,
    desc: str,
    ndcg5_a: np.ndarray,
    ndcg10_a: np.ndarray,
    p10_a: np.ndarray,
    ndcg5_b: np.ndarray,
    ndcg10_b: np.ndarray,
    p10_b: np.ndarray,
    n_permutations: int = 10000,
) -> dict:
    n_total = len(ndcg10_a)
    n_eff = int(np.sum(ndcg10_a != ndcg10_b))
    mean_a = float(np.mean(ndcg10_a))
    mean_b = float(np.mean(ndcg10_b))
    delta = mean_a - mean_b

    # ── Wilcoxon signed-rank (pratt) ──────────────────────────────────────────
    try:
        w_stat, p_w = wilcoxon(ndcg10_a, ndcg10_b, alternative="two-sided", zero_method="pratt")
    except ValueError:
        # All values identical
        w_stat, p_w = 0.0, 1.0

    # ── Permutation test (vectorised) ─────────────────────────────────────────
    def _stat(x, y):
        return np.mean(x) - np.mean(y)

    perm_result = permutation_test(
        (ndcg10_a, ndcg10_b),
        _stat,
        permutation_type="samples",
        n_resamples=n_permutations,
        alternative="two-sided",
        random_state=SEED,
    )
    p_perm = float(perm_result.pvalue)

    # ── Effect size: rank-biserial r ──────────────────────────────────────────
    # Zero differences are dropped (see significance.rank_biserial)
    r = rank_biserial(ndcg10_a, ndcg10_b)

    # ── Secondary metrics (nDCG@5, P@10) ─────────────────────────────────────
    try:
        _, p_w5 = wilcoxon(ndcg5_a, ndcg5_b, alternative="two-sided", zero_method="pratt")
        _, p_wp = wilcoxon(p10_a, p10_b, alternative="two-sided", zero_method="pratt")
    except ValueError:
        p_w5, p_wp = 1.0, 1.0

    perm5 = permutation_test(
        (ndcg5_a, ndcg5_b),
        _stat,
        permutation_type="samples",
        n_resamples=n_permutations,
        alternative="two-sided",
        random_state=SEED,
    )
    permp = permutation_test(
        (p10_a, p10_b),
        _stat,
        permutation_type="samples",
        n_resamples=n_permutations,
        alternative="two-sided",
        random_state=SEED,
    )

    return {
        "pair_id": pair_id,
        "family": family,
        "description": desc,
        "method_a": method_a,
        "method_b": method_b,
        "n_users_total": n_total,
        "n_efetivo": n_eff,
        "mean_ndcg10_a": round(mean_a, 4),
        "mean_ndcg10_b": round(mean_b, 4),
        "delta_ndcg10": round(delta, 4),
        "wilcoxon_stat": round(w_stat, 2),
        "p_wilcoxon_raw": p_w,
        "p_wilcoxon_holm": None,  # filled in once the whole family is collected
        "p_perm_raw": p_perm,
        "p_perm_holm": None,
        "effect_size_r": round(r, 4),
        "effect_size_label": effect_label(r),
        "significant_wilcoxon_holm": None,
        "significant_perm_holm": None,
        # secondary metrics
        "p_wilcoxon_ndcg5_raw": round(p_w5, 6),
        "p_wilcoxon_p10_raw": round(p_wp, 6),
        "p_perm_ndcg5_raw": round(float(perm5.pvalue), 6),
        "p_perm_p10_raw": round(float(permp.pvalue), 6),
    }


# ─── Per-user CSV loading ────────────────────────────────────────────────────


def load_csv(label: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    path = ST_DIR / f"per_user_{label}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"CSV not found: {path}\nRun acw.evaluation.per_user_metrics first."
        )
    ndcg5, ndcg10, p10 = [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            ndcg5.append(float(row["ndcg5"]))
            ndcg10.append(float(row["ndcg10"]))
            p10.append(float(row["p10"]))
    return np.array(ndcg5), np.array(ndcg10), np.array(p10)


# ─── Validation users for Family 2 ───────────────────────────────────────────


def get_val_users(n_permutations: int) -> list[str]:
    """Return the canonical validation users.

    Loads canonical_val_users.json (see generate_canonical_split). If it is
    missing, rebuilds a seeded 80/20 split of the H3-indexed eligible users
    and returns the last 20%.
    """
    import json

    from acw.evaluation.config import resolve_paths
    from acw.evaluation.loader import build_genre_ground_truth

    cfg = resolve_paths("music")
    val_path = cfg.canonical_val_path

    if val_path.exists():
        with open(val_path) as f:
            return json.load(f)

    # Fallback: seeded 80/20 split
    genre_gt = build_genre_ground_truth(cfg)
    with open(ROOT / "data" / "processed" / "user_profile_index_h3.json") as f:
        h3_idx: dict[str, int] = json.load(f)

    eligible = [u for u in sorted(h3_idx, key=lambda x: h3_idx[x]) if u in genre_gt]
    rng = np.random.default_rng(SEED)
    idxs = np.arange(len(eligible))
    rng.shuffle(idxs)
    n_train = int(len(idxs) * 0.8)
    return [eligible[i] for i in idxs[n_train:]]


def compute_val_metrics(
    label: str,
    npy_rel: str,
    val_users: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute per-user nDCG@5, nDCG@10 and P@10 on ``val_users`` against whitened album embeddings."""
    import json

    from acw.evaluation.config import resolve_paths
    from acw.evaluation.loader import build_genre_ground_truth
    from acw.evaluation.metrics import ndcg_at_k, precision_at_k

    cfg = resolve_paths("music")
    genre_gt = build_genre_ground_truth(cfg)

    PROCESSED = ROOT / "data" / "processed"
    ALIGNMENT = PROCESSED / "alignment"
    items_emb = np.load(ALIGNMENT / "album_embeddings_whitened.npy").astype(np.float32)
    profiles = np.load(PROCESSED / npy_rel).astype(np.float32)

    with open(PROCESSED / "album_id_to_idx.json") as f:
        item_id_to_idx: dict[str, int] = json.load(f)
    idx_to_item = {v: k for k, v in item_id_to_idx.items()}

    with open(PROCESSED / "user_profile_index_h3.json") as f:
        h3_idx: dict[str, int] = json.load(f)

    ndcg5_l, ndcg10_l, p10_l = [], [], []
    for uid in val_users:
        rel = genre_gt.get(uid)
        if not rel:
            continue
        row = h3_idx.get(uid)
        if row is None:
            continue
        sims = items_emb @ profiles[row]
        top_indices = np.argsort(sims)[::-1][:20]
        ranked = [idx_to_item[i] for i in top_indices]
        ndcg5_l.append(ndcg_at_k(ranked, rel, 5))
        ndcg10_l.append(ndcg_at_k(ranked, rel, 10))
        p10_l.append(precision_at_k(ranked, rel, 10))

    return np.array(ndcg5_l), np.array(ndcg10_l), np.array(p10_l)


# ─── Main ─────────────────────────────────────────────────────────────────────


def main(n_permutations: int = 10000):
    log.info("=" * 70)
    log.info("Statistical significance tests")
    log.info(f"  Wilcoxon(pratt) + permutation_test(n={n_permutations}) + Holm")
    log.info("=" * 70)

    all_rows: list[dict] = []

    # ── SANITY CHECKS ─────────────────────────────────────────────────────────
    log.info("\n── Sanity checks (no Holm correction) ────────────────────────────")
    for pair_id, ma, mb, desc in SANITY_CHECKS:
        log.info(f"  {pair_id}: {ma} vs {mb}")
        n5a, n10a, p10a = load_csv(ma)
        n5b, n10b, p10b = load_csv(mb)

        # Truncate to a common length (Random uses the H2 index, same size)
        n = min(len(n10a), len(n10b))
        row = run_pair(
            pair_id,
            "sanity",
            ma,
            mb,
            desc,
            n5a[:n],
            n10a[:n],
            p10a[:n],
            n5b[:n],
            n10b[:n],
            p10b[:n],
            n_permutations,
        )
        # Sanity checks: no Holm correction, use raw p-values
        row["p_wilcoxon_holm"] = row["p_wilcoxon_raw"]
        row["p_perm_holm"] = row["p_perm_raw"]
        row["significant_wilcoxon_holm"] = bool(row["p_wilcoxon_raw"] < 0.05)
        row["significant_perm_holm"] = bool(row["p_perm_raw"] < 0.05)

        log.info(
            f"    nDCG@10: {row['mean_ndcg10_a']:.4f} vs {row['mean_ndcg10_b']:.4f}  "
            f"Δ={row['delta_ndcg10']:+.4f}  "
            f"p_wilcoxon={row['p_wilcoxon_raw']:.2e}  "
            f"p_perm={row['p_perm_raw']:.2e}  "
            f"r={row['effect_size_r']:.3f} ({row['effect_size_label']})"
        )

        if not row["significant_wilcoxon_holm"]:
            log.error(
                f"  {pair_id} SANITY CHECK FAILED: {ma} does not beat {mb} (p={row['p_wilcoxon_raw']:.4f})"
            )
        all_rows.append(row)

    # ── FAMILY 1 ──────────────────────────────────────────────────────────────
    log.info(f"\n── Family 1 (k={len(FAMILY_1)}) ──────────────────────────────────────")
    f1_rows: list[dict] = []
    for pair_id, ma, mb, desc in FAMILY_1:
        log.info(f"  {pair_id}: {ma} vs {mb}")
        n5a, n10a, p10a = load_csv(ma)
        n5b, n10b, p10b = load_csv(mb)
        n = min(len(n10a), len(n10b))
        row = run_pair(
            pair_id,
            "familia_1",
            ma,
            mb,
            desc,
            n5a[:n],
            n10a[:n],
            p10a[:n],
            n5b[:n],
            n10b[:n],
            p10b[:n],
            n_permutations,
        )
        log.info(
            f"    nDCG@10: {row['mean_ndcg10_a']:.4f} vs {row['mean_ndcg10_b']:.4f}  "
            f"Δ={row['delta_ndcg10']:+.4f}  "
            f"p_wilcoxon_raw={row['p_wilcoxon_raw']:.2e}  "
            f"p_perm_raw={row['p_perm_raw']:.2e}  "
            f"r={row['effect_size_r']:.3f} ({row['effect_size_label']})  "
            f"n_eff={row['n_efetivo']}"
        )
        f1_rows.append(row)

    holm_rows(f1_rows, "p_wilcoxon_raw", "p_wilcoxon_holm", "significant_wilcoxon_holm")
    holm_rows(f1_rows, "p_perm_raw", "p_perm_holm", "significant_perm_holm")

    log.info("\n  Holm-adjusted results (Family 1):")
    for row in f1_rows:
        sig_w = "✓" if row["significant_wilcoxon_holm"] else "✗"
        sig_p = "✓" if row["significant_perm_holm"] else "✗"
        log.info(
            f"    {row['pair_id']} {row['method_a']} vs {row['method_b']}: "
            f"p_holm_wilcoxon={row['p_wilcoxon_holm']:.4f}{sig_w}  "
            f"p_holm_perm={row['p_perm_holm']:.4f}{sig_p}  "
            f"r={row['effect_size_r']:.3f}"
        )
    all_rows.extend(f1_rows)

    # ── FAMILY 2 ──────────────────────────────────────────────────────────────
    log.info(f"\n── Family 2 (k={len(FAMILY_2)}, val split) ───────────────────────────")
    log.info("  Loading validation users...")
    val_users = get_val_users(n_permutations)
    log.info(f"  Val users: {len(val_users)}")

    # Label → profile .npy (relative to data/processed/)
    _val_npys = {
        "H3-NL-PW-ZB": "alignment/user_profiles_h3_nlpw_zb.npy",
        "H3-PW-ZB-val": "alignment/user_profiles_h3_pw_zb.npy",
        "H3-MoE-ZB": "alignment/user_profiles_h3_moe_zb.npy",
        "H3-MoE-PW": "alignment/user_profiles_h3_moe.npy",
    }

    f2_data: dict[str, tuple] = {}
    for lbl, rel in _val_npys.items():
        path = ROOT / "data" / "processed" / rel
        if not path.exists():
            log.warning(f"  {lbl}: {path} not found — skipping in Family 2")
            f2_data[lbl] = (np.array([]), np.array([]), np.array([]))
        else:
            f2_data[lbl] = compute_val_metrics(lbl, rel, val_users)
            log.info(f"  {lbl}: nDCG@10={np.mean(f2_data[lbl][1]):.4f}  n={len(f2_data[lbl][1])}")

    f2_rows: list[dict] = []
    for pair_id, ma, mb, desc in FAMILY_2:
        if len(f2_data.get(ma, ([], [], []))[1]) == 0 or len(f2_data.get(mb, ([], [], []))[1]) == 0:
            log.warning(f"  {pair_id}: missing data — skipping")
            continue
        log.info(f"  {pair_id}: {ma} vs {mb}")
        n5a, n10a, p10a = f2_data[ma]
        n5b, n10b, p10b = f2_data[mb]
        n = min(len(n10a), len(n10b))
        row = run_pair(
            pair_id,
            "familia_2",
            ma,
            mb,
            desc,
            n5a[:n],
            n10a[:n],
            p10a[:n],
            n5b[:n],
            n10b[:n],
            p10b[:n],
            n_permutations,
        )
        log.info(
            f"    nDCG@10: {row['mean_ndcg10_a']:.4f} vs {row['mean_ndcg10_b']:.4f}  "
            f"Δ={row['delta_ndcg10']:+.4f}  "
            f"p_wilcoxon_raw={row['p_wilcoxon_raw']:.2e}  "
            f"r={row['effect_size_r']:.3f} ({row['effect_size_label']})"
        )
        f2_rows.append(row)

    if f2_rows:
        holm_rows(f2_rows, "p_wilcoxon_raw", "p_wilcoxon_holm", "significant_wilcoxon_holm")
        holm_rows(f2_rows, "p_perm_raw", "p_perm_holm", "significant_perm_holm")
    all_rows.extend(f2_rows)

    # ── Write full CSV ────────────────────────────────────────────────────────
    out_path = ST_DIR / "significance_results.csv"
    fieldnames = [
        "pair_id",
        "family",
        "description",
        "method_a",
        "method_b",
        "n_users_total",
        "n_efetivo",
        "mean_ndcg10_a",
        "mean_ndcg10_b",
        "delta_ndcg10",
        "wilcoxon_stat",
        "p_wilcoxon_raw",
        "p_wilcoxon_holm",
        "p_perm_raw",
        "p_perm_holm",
        "effect_size_r",
        "effect_size_label",
        "significant_wilcoxon_holm",
        "significant_perm_holm",
        "p_wilcoxon_ndcg5_raw",
        "p_wilcoxon_p10_raw",
        "p_perm_ndcg5_raw",
        "p_perm_p10_raw",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    log.info(f"\n  Results saved: {out_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — significance tests")
    log.info("=" * 70)
    log.info(
        f"  {'Pair':<4} {'Family':<12} {'Method A':<16} {'Method B':<16} "
        f"{'Δ nDCG@10':>10} {'p_holm_W':>10} {'p_holm_P':>10} "
        f"{'r':>6} {'sig':>4}"
    )
    log.info("  " + "─" * 90)
    for row in all_rows:
        sig = (
            "✓"
            if (row["significant_wilcoxon_holm"] and row["significant_perm_holm"])
            else "~"
            if (row["significant_wilcoxon_holm"] or row["significant_perm_holm"])
            else "✗"
        )
        log.info(
            f"  {row['pair_id']:<4} {row['family']:<12} "
            f"{row['method_a']:<16} {row['method_b']:<16} "
            f"{row['delta_ndcg10']:>+10.4f} "
            f"{row['p_wilcoxon_holm']:>10.4f} "
            f"{row['p_perm_holm']:>10.4f} "
            f"{row['effect_size_r']:>6.3f} {sig:>4}"
        )
    log.info("=" * 70)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--n-permutations", type=int, default=10000)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(n_permutations=args.n_permutations)
