"""Significance tests for the auxiliary domains (movies, games).

One family per domain (k=14): for each heuristic h ∈ {H3, H4}, the 7
whitened/aligned methods ("{h} (vs whitened)", "{h}+PW", "{h}+PW-ZB",
"{h}+NL-PW", "{h}+NL-PW-ZB", "{h}+MoE-PW", "{h}+MoE-ZB") are compared
against the unsupervised baseline "H2 (vs whitened)": does learned alignment
beat asymmetric whitening in these domains?

Wilcoxon signed-rank (zero_method='pratt') + vectorised permutation test
(n=10000, seed=42) + Holm-Bonferroni correction per domain.

Pairs are aligned on the INTERSECTION of user_ids rather than by position,
because H3/H4 methods cover different user subsets than the H2 baseline.

Usage
-----
    python -m acw.evaluation.significance_tests_aux_domains
"""

from __future__ import annotations

import csv
import json
import logging
import sys

import numpy as np
from scipy.stats import permutation_test, wilcoxon

from acw.evaluation.config import KNOWN_METHODS, PROCESSED, resolve_paths
from acw.evaluation.loader import build_genre_ground_truth, load_profiles, sample_test_users
from acw.evaluation.metrics import ndcg_at_k, precision_at_k
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
N_PERMUTATIONS = 10000

BASELINE = "H2 (vs whitened)"
FAMILY_LABELS = [
    "{h} (vs whitened)",
    "{h}+PW",
    "{h}+PW-ZB",
    "{h}+NL-PW",
    "{h}+NL-PW-ZB",
    "{h}+MoE-PW",
    "{h}+MoE-ZB",
]


def compute_per_user_ndcg10(
    domain: str,
    label: str,
    genre_gt: dict[str, set[str]],
    test_users: list[str],
    idx_to_item: dict[int, str],
    default_items_emb: np.ndarray,
) -> dict[str, tuple[float, float, float]]:
    """Return ``{user_id: (ndcg5, ndcg10, p10)}`` for a method in KNOWN_METHODS."""
    entry = next((e for e in KNOWN_METHODS.get(domain, []) if e[0] == label), None)
    if entry is None:
        raise ValueError(f"Method {label!r} is not registered for domain={domain}")
    _, rel_path, idx_rel, emb_rel, _ = entry

    path = PROCESSED / rel_path
    profiles = np.load(path).astype(np.float32)
    if idx_rel:
        with open(PROCESSED / idx_rel) as f:
            idx_map: dict[str, int] = json.load(f)
    else:
        _, idx_map = load_profiles(path)

    items_emb = np.load(PROCESSED / emb_rel).astype(np.float32) if emb_rel else default_items_emb

    max_k = 20
    out: dict[str, tuple[float, float, float]] = {}
    for uid in test_users:
        relevant = genre_gt.get(uid)
        if not relevant:
            continue
        row = idx_map.get(uid)
        if row is None:
            continue
        sims = items_emb @ profiles[row]
        top_indices = np.argsort(sims)[::-1][:max_k]
        ranked = [idx_to_item[i] for i in top_indices]
        out[uid] = (
            ndcg_at_k(ranked, relevant, 5),
            ndcg_at_k(ranked, relevant, 10),
            precision_at_k(ranked, relevant, 10),
        )
    return out


def save_per_user_csv(domain: str, label: str, data: dict[str, tuple]) -> None:
    safe_label = label.replace(" ", "_").replace("(", "").replace(")", "")
    path = ST_DIR / f"per_user_{domain}_{safe_label}.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "ndcg5", "ndcg10", "p10"])
        for uid, (n5, n10, p10) in data.items():
            writer.writerow([uid, n5, n10, p10])


def run_pair(
    pair_id: str,
    domain: str,
    method_a: str,
    method_b: str,
    data_a: dict[str, tuple],
    data_b: dict[str, tuple],
) -> dict:
    common = sorted(set(data_a) & set(data_b))
    n_common = len(common)
    ndcg10_a = np.array([data_a[u][1] for u in common])
    ndcg10_b = np.array([data_b[u][1] for u in common])

    mean_a, mean_b = float(np.mean(ndcg10_a)), float(np.mean(ndcg10_b))
    delta = mean_a - mean_b

    try:
        _, p_w = wilcoxon(ndcg10_a, ndcg10_b, alternative="two-sided", zero_method="pratt")
    except ValueError:
        p_w = 1.0

    def _stat(x, y):
        return np.mean(x) - np.mean(y)

    perm = permutation_test(
        (ndcg10_a, ndcg10_b),
        _stat,
        permutation_type="samples",
        n_resamples=N_PERMUTATIONS,
        alternative="two-sided",
        random_state=SEED,
    )
    p_perm = float(perm.pvalue)

    r = rank_biserial(ndcg10_a, ndcg10_b)

    return {
        "pair_id": pair_id,
        "domain": domain,
        "method_a": method_a,
        "method_b": method_b,
        "n_common": n_common,
        "mean_ndcg10_a": round(mean_a, 4),
        "mean_ndcg10_b": round(mean_b, 4),
        "delta_ndcg10": round(delta, 4),
        "p_wilcoxon_raw": p_w,
        "p_wilcoxon_holm": None,
        "p_perm_raw": p_perm,
        "p_perm_holm": None,
        "effect_size_r": round(r, 4),
        "effect_size_label": effect_label(r),
        "significant_wilcoxon_holm": None,
        "significant_perm_holm": None,
    }


def main() -> None:
    all_rows: list[dict] = []

    for domain in ["movies", "games"]:
        log.info("=" * 70)
        log.info(f"Domain: {domain}")
        log.info("=" * 70)

        cfg = resolve_paths(domain=domain)
        default_items_emb = np.load(cfg.items_emb_path).astype(np.float32)
        with open(cfg.items_idx_path) as f:
            item_id_to_idx: dict[str, int] = json.load(f)
        idx_to_item = {v: k for k, v in item_id_to_idx.items()}

        genre_gt = build_genre_ground_truth(cfg)
        test_users = sample_test_users(genre_gt, cfg)
        log.info(f"  Test users: {len(test_users):,}")

        labels_needed = {BASELINE} | {
            tmpl.format(h=h) for tmpl in FAMILY_LABELS for h in ["H3", "H4"]
        }
        per_user_data: dict[str, dict[str, tuple]] = {}
        for label in labels_needed:
            data = compute_per_user_ndcg10(
                domain, label, genre_gt, test_users, idx_to_item, default_items_emb
            )
            per_user_data[label] = data
            save_per_user_csv(domain, label, data)
            log.info(
                f"  {label:<28}: n={len(data)}  nDCG@10={np.mean([v[1] for v in data.values()]):.4f}"
            )

        domain_rows: list[dict] = []
        pair_n = 1
        for h in ["H3", "H4"]:
            for tmpl in FAMILY_LABELS:
                label = tmpl.format(h=h)
                pair_id = f"{domain[:1].upper()}{pair_n}"
                pair_n += 1
                row = run_pair(
                    pair_id,
                    domain,
                    label,
                    BASELINE,
                    per_user_data[label],
                    per_user_data[BASELINE],
                )
                log.info(
                    f"    {pair_id} {label:<16} vs {BASELINE}: "
                    f"Δ={row['delta_ndcg10']:+.4f}  n={row['n_common']}  "
                    f"p_w={row['p_wilcoxon_raw']:.2e}  p_perm={row['p_perm_raw']:.2e}  "
                    f"r={row['effect_size_r']:.3f} ({row['effect_size_label']})"
                )
                domain_rows.append(row)

        holm_rows(domain_rows, "p_wilcoxon_raw", "p_wilcoxon_holm", "significant_wilcoxon_holm")
        holm_rows(domain_rows, "p_perm_raw", "p_perm_holm", "significant_perm_holm")

        log.info(f"\n  Holm-adjusted results ({domain}, k={len(domain_rows)}):")
        for row in domain_rows:
            sig_w = "✓" if row["significant_wilcoxon_holm"] else "✗"
            sig_p = "✓" if row["significant_perm_holm"] else "✗"
            log.info(
                f"    {row['pair_id']} {row['method_a']:<16} Δ={row['delta_ndcg10']:+.4f} "
                f"p_holm_w={row['p_wilcoxon_holm']:.4f}{sig_w} "
                f"p_holm_perm={row['p_perm_holm']:.4f}{sig_p} "
                f"r={row['effect_size_r']:.3f}"
            )
        all_rows.extend(domain_rows)

    out_path = ST_DIR / "significance_results_movies_games.csv"
    fieldnames = [
        "pair_id",
        "domain",
        "method_a",
        "method_b",
        "n_common",
        "mean_ndcg10_a",
        "mean_ndcg10_b",
        "delta_ndcg10",
        "p_wilcoxon_raw",
        "p_wilcoxon_holm",
        "p_perm_raw",
        "p_perm_holm",
        "effect_size_r",
        "effect_size_label",
        "significant_wilcoxon_holm",
        "significant_perm_holm",
    ]
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)
    log.info(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
