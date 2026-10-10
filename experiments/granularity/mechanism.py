"""
Mechanism diagnostics for profile-side vs catalog-side ZCA (test users).

For one (domain, encoder, titlecat) it reuses the encoder REGISTRY, the H2
profiles and the correction functions of run_corrections / acw.corrections and
reports, on the TEST users of results/granularity/{domain}_{encoder}_titlecat.csv.gz
(the evaluated population; the CSV is also used to check that the rankings
rebuilt here reproduce its per-user nDCG@10):

  A  geometry of profiles (raw vs zca_profile) and of the catalog (raw vs
     zca_catalog_a1.0 vs center_catalog): mean pairwise cosine (exact closed form
     over all vectors, they are unit norm), effective rank (exp of the entropy of
     the normalised covariance eigenvalues), variance share of PC1 / top-10 PCs.
  B  retrieval diversity of the top-10 lists of raw, zca_catalog_a1.0,
     zca_profile, zca_both_separate, popularity and center_profile (the profile
     centred with the mean train profile, no whitening; the per-user check and the
     depth-2 scores come from query_norm_{domain}_{encoder}.csv.gz, arm
     center_train_users) over <= MAX_USERS sampled test
     users: catalog coverage (% distinct items), Gini of item exposure (over the
     whole catalog), mean Jaccard of the lists of N_PAIRS random user pairs,
     k-occurrence skewness (k=10, over the whole catalog), mean popularity
     percentile (rank of the number of train users who consumed the item,
     among catalog items) of recommended items.
  C  category structure of the catalog restricted to items with a depth-2
     category (transforms fitted on the full catalog): trace between/within
     category variance ratio and the share of each item's 10 nearest neighbours
     (among those items) with its category, for N_NN_ITEMS sampled items.
  D  Spearman correlation across test users between the raw-profile typicality
     (cosine to the mean train profile) and the nDCG@10 gain zca_profile - raw
     (exact ground truth, from the per-user CSV).
  E  category concentration of the top-10 lists (same sampled users as B; raw,
     zca_catalog_a1.0, zca_profile, zca_both_separate, popularity, center_profile,
     top_category;
     depth-2 categories): Shannon entropy (bits) of the pooled recommendations'
     category distribution and their share in the 1 / 5 categories reached by
     most TRAIN users; per user, the Jaccard overlap between the top-10 list and
     the constant top_category list and its Spearman correlation across users
     with the user's depth-2 nDCG@10 (per condition and pooled over conditions;
     undefined, stored as null, for the two constant-list conditions).

--profile-only computes just A (profiles). --music-encoders aggregates the
music results/mechanism/music_{encoder}.json files and correlates raw-profile
geometry with the zca_profile exact-GT relative gain of summary_test.csv.

Usage:
    python -m experiments.granularity.mechanism --domain music --encoder gte
    python -m experiments.granularity.mechanism --domain music --encoder bge-m3 --profile-only
    python -m experiments.granularity.mechanism --music-encoders
"""

from __future__ import annotations

import argparse
import json
import logging
import resource
import time

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import rankdata, skew, spearmanr

from acw.corrections import abtt, center, fit_moments, zca_power
from acw.data.splits import load_split, split_path
from acw.embeddings.generate_h2_profiles.io import load_user_history, user_history_path
from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_ground_truth, item_genres
from acw.evaluation.pipeline import (
    MAX_K,
    train_category_counts,
    train_popularity,
    trivial_rankings,
)
from acw.evaluation.ranking import top_k
from acw.provenance import provenance
from acw.settings import PROCESSED_DIR, RESULTS_DIR, SEED
from experiments.granularity.run_corrections import (
    OUT_DIR,
    REGISTRY,
    _load_matrix,
    _row,
    category_levels,
    h2_profiles,
)

log = logging.getLogger("mechanism")

MECH_DIR = RESULTS_DIR / "mechanism"
MAX_USERS = 3000  # users sampled for the list-level diversity metrics (section B)
N_PAIRS = 2000  # random user pairs for the mean Jaccard overlap
N_NN_ITEMS = 5000  # items sampled for the neighbour-category share (section C)
TOP = 10
CAT_DEPTH = 2
CHUNK = 20_000  # rows per float64 block
B_CONDITIONS = (
    "raw",
    "zca_catalog_a1.0",
    "zca_profile",
    "zca_both_separate",
    "popularity",
    "center_profile",
)
# the pooled correlation of section E keeps its original four conditions
E_PERSONALISED = ("raw", "zca_catalog_a1.0", "zca_profile", "zca_both_separate")
E_CONDITIONS = (*B_CONDITIONS, "top_category")
QN_CENTER = "center_train_users"  # arm of query_norm.py that center_profile reproduces


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────


def geometry(x: np.ndarray) -> dict:
    """Geometry of a set of unit-norm vectors (n, d)."""
    n = len(x)
    s = x.sum(0, dtype=np.float64)
    sq = float(np.einsum("ij,ij->", x, x, dtype=np.float64))
    ev = fit_moments(x).eigvals
    p = ev[ev > 0] / ev.sum()
    return {
        "n": n,
        "mean_pairwise_cosine": float((s @ s - sq) / (n * (n - 1))),
        "effective_rank": float(np.exp(-(p * np.log(p)).sum())),
        "var_share_pc1": float(ev[0] / ev.sum()),
        "var_share_top10": float(ev[:10].sum() / ev.sum()),
    }


def gini(counts: np.ndarray) -> float:
    c = np.sort(counts.astype(np.float64))
    n = len(c)
    return float(2 * (np.arange(1, n + 1) @ c) / (n * c.sum()) - (n + 1) / n)


def diversity(top: np.ndarray, n_items: int, pop_pct: np.ndarray, pairs: np.ndarray) -> dict:
    """Diversity of (n_users, TOP) lists of item rows; ``pairs`` is (n_pairs, 2) user indices."""
    counts = np.bincount(top.ravel(), minlength=n_items)
    a, b = top[pairs[:, 0]], top[pairs[:, 1]]
    inter = (a[:, :, None] == b[:, None, :]).sum((1, 2))  # lists hold distinct items
    return {
        "coverage_pct": float(100.0 * (counts > 0).mean()),
        "gini_exposure": gini(counts),
        "mean_jaccard": float((inter / (2 * top.shape[1] - inter)).mean()),
        "hubness_skewness_k10": float(skew(counts)),
        "mean_popularity_percentile": float(pop_pct[top].mean()),
    }


def between_within(x: np.ndarray, codes: np.ndarray) -> float:
    """Trace of the between-category scatter over the within-category scatter."""
    n, g = len(x), int(codes.max()) + 1
    sums = np.zeros((g, x.shape[1]))
    for i in range(0, n, CHUNK):
        sl = codes[i : i + CHUNK]
        ind = sparse.csr_matrix((np.ones(len(sl)), (sl, np.arange(len(sl)))), shape=(g, len(sl)))
        sums += ind @ x[i : i + CHUNK].astype(np.float64)
    sq = float(np.einsum("ij,ij->", x, x, dtype=np.float64))
    grp = float(((sums**2).sum(1) / np.bincount(codes, minlength=g)).sum())
    mu = sums.sum(0) / n
    return (grp - n * float(mu @ mu)) / (sq - grp)


def neighbour_same_category(x: np.ndarray, codes: np.ndarray, sample: np.ndarray) -> float:
    nb = top_k(x[sample], x, TOP + 1)
    order = np.argsort(nb == sample[:, None], axis=1, kind="stable")[:, :TOP]  # drop self
    nb = np.take_along_axis(nb, order, axis=1)
    return float((codes[nb] == codes[sample][:, None]).mean())


def category_concentration(cat: np.ndarray, train_top: list[int], n_cat: int) -> dict:
    """Concentration of pooled recommendations' category codes ``cat`` (-1 =
    uncategorised); ``train_top`` lists the most frequent train-user categories."""
    known = cat[cat >= 0]
    p = np.bincount(known, minlength=n_cat) / len(known)
    p = p[p > 0]
    return {
        "entropy_bits": float((p * np.log2(1 / p)).sum()),
        "n_distinct_categories": int(len(p)),
        "share_top1_train_category": float(np.isin(known, train_top[:1]).mean()),
        "share_top5_train_categories": float(np.isin(known, train_top[:5]).mean()),
        "share_uncategorised": float((cat < 0).mean()),
    }


def jaccard_to_list(top: np.ndarray, const: np.ndarray) -> np.ndarray:
    """Per-row Jaccard overlap between (n, k) lists of distinct items and one list."""
    inter = (top[:, :, None] == const[None, None, :]).sum((1, 2))
    return inter / (top.shape[1] + len(const) - inter)


def spearman(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rho and p; null when either input is constant."""
    if np.ptp(x) == 0 or np.ptp(y) == 0:
        return {"rho": None, "p": None}
    rho, p = spearmanr(x, y)
    return {"rho": float(rho), "p": float(p)}


# ─────────────────────────────────────────────────────────────────────────────
# Cross-encoder aggregation
# ─────────────────────────────────────────────────────────────────────────────


def music_encoders() -> None:
    summ = pd.read_csv(OUT_DIR / "summary_test.csv")
    summ = summ[
        (summ.domain == "music") & (summ.text == "titlecat") & (summ.granularity == "exact")
    ]
    rows = {}
    inputs = [OUT_DIR / "summary_test.csv"]
    for enc in sorted(e for d, e, t in REGISTRY if d == "music" and t == "titlecat"):
        inputs.append(MECH_DIR / f"music_{enc}.json")
        g = json.loads(inputs[-1].read_text())["A"]["profiles"]["raw"]
        r = summ[(summ.encoder == enc) & (summ.condition == "zca_profile")].iloc[0]
        rows[enc] = {
            "raw_profile_mean_pairwise_cosine": g["mean_pairwise_cosine"],
            "raw_profile_effective_rank": g["effective_rank"],
            "zca_profile_exact_relative_gain": float(r.ndcg10_delta / (r.ndcg10 - r.ndcg10_delta)),
        }
    df = pd.DataFrame(rows).T
    out = {"n_encoders": len(df), "per_encoder": rows, "provenance": provenance(inputs)}
    for col in ("raw_profile_mean_pairwise_cosine", "raw_profile_effective_rank"):
        rho, p = spearmanr(df[col], df["zca_profile_exact_relative_gain"])
        out[f"spearman_{col}_vs_gain"] = {"rho": float(rho), "p": float(p)}
    (MECH_DIR / "music_encoders.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=["music", "movies", "games"])
    ap.add_argument("--encoder")
    ap.add_argument("--profile-only", action="store_true")
    ap.add_argument("--music-encoders", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    MECH_DIR.mkdir(parents=True, exist_ok=True)
    if args.music_encoders:
        return music_encoders()
    key = (args.domain, args.encoder, "titlecat")
    if key not in REGISTRY:
        raise SystemExit(f"no registry entry for {key}")
    t0 = time.time()
    src, cfg = REGISTRY[key], resolve_paths(args.domain)
    csv = OUT_DIR / f"{args.domain}_{args.encoder}_titlecat.csv.gz"
    if not csv.exists():
        raise SystemExit(f"{csv} missing: run experiments.granularity.run_corrections first")
    per_user = pd.read_csv(csv, usecols=["user_id", "split", "gt", "condition", "nDCG@10"])
    per_user = per_user[per_user["split"] == "test"]

    def pivot(gt: str) -> pd.DataFrame:
        sub = per_user[per_user["gt"] == gt]  # .gt is a method
        return sub.pivot(index="user_id", columns="condition", values="nDCG@10")

    ndcg, ndcg_d2 = pivot("exact"), pivot("d2")
    qn_csv = OUT_DIR / f"query_norm_{args.domain}_{args.encoder}.csv.gz"
    if not args.profile_only:
        if not qn_csv.exists():
            raise SystemExit(f"{qn_csv} missing: run experiments.granularity.query_norm first")
        qn = pd.read_csv(qn_csv, usecols=["user_id", "split", "gt", "condition", "nDCG@10"])
        qn = qn[(qn["split"] == "test") & (qn["condition"] == QN_CENTER)]
        for gt, tab in (("exact", ndcg), ("d2", ndcg_d2)):
            tab["center_profile"] = qn[qn["gt"] == gt].set_index("user_id")["nDCG@10"]
    test_users = list(
        per_user.loc[(per_user["condition"] == "raw") & (per_user["gt"] == "exact"), "user_id"]
    )

    books, book_idx = _load_matrix(src.books, src.books_idx)
    history = load_user_history(args.domain)
    train_users = load_split(args.domain, "train")
    train_prof, _, _ = h2_profiles(train_users, history, books, book_idx)
    test_prof, kept, _ = h2_profiles(test_users, history, books, book_idx)
    assert kept == test_users
    del books
    mom_p = fit_moments(train_prof)
    zca_test = zca_power(test_prof, *mom_p, 1.0)
    center_test = center(test_prof, mom_p.mean)
    log.info(f"train profiles {train_prof.shape}, test profiles {test_prof.shape}")

    A = {"profiles": {"raw": geometry(test_prof), "zca_profile": geometry(zca_test)}}
    out = {
        "domain": args.domain,
        "encoder": args.encoder,
        "n_test_users": len(test_users),
        "n_train_profiles": len(train_prof),
        "A": A,
    }
    if not args.profile_only:
        out |= run_catalog_sections(
            cfg, src, test_users, train_users, train_prof, test_prof, zca_test, ndcg, ndcg_d2, A,
            center_test,
        )  # fmt: skip
    inputs = [
        csv,
        *([qn_csv] if not args.profile_only else []),
        PROCESSED_DIR / src.books,
        PROCESSED_DIR / src.books_idx,
        user_history_path(args.domain),
        split_path(args.domain, "train"),
    ]
    if not args.profile_only:
        inputs += [PROCESSED_DIR / src.items, PROCESSED_DIR / src.items_idx]
        inputs += [cfg.items_idx_path, cfg.items_meta_path, cfg.gt_path]
    out["provenance"] = provenance(inputs)
    out["runtime_s"] = round(time.time() - t0, 1)
    out["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    path = MECH_DIR / f"{args.domain}_{args.encoder}.json"
    path.write_text(json.dumps(out, indent=2))
    log.info(f"saved {path}  runtime {out['runtime_s']}s  peak RSS {out['peak_rss_mb']} MB")


def run_catalog_sections(
    cfg,
    src,
    test_users,
    train_users,
    train_prof,
    test_prof,
    zca_test,
    ndcg,
    ndcg_d2,
    A,
    center_test,
) -> dict:
    items, item_idx = _load_matrix(src.items, src.items_idx)
    idx_to_item = {v: k for k, v in item_idx.items()}
    n_items = len(items)
    mom_c = fit_moments(items)
    zca_items = zca_power(items, *mom_c, 1.0)
    A["catalog"] = {
        "raw": geometry(items),
        "zca_catalog_a1.0": geometry(zca_items),
        "center_catalog": geometry(center(items, mom_c.mean)),
    }
    log.info("A done")

    # Rankings of the B conditions for all test users (also checked against the CSV)
    const_lists = trivial_rankings(cfg, idx_to_item, train_users, depth=CAT_DEPTH)[0]
    ranks = {
        "raw": top_k(test_prof, items, MAX_K),
        "zca_catalog_a1.0": top_k(test_prof, zca_items, MAX_K),
        "zca_profile": top_k(zca_test, items, MAX_K),
        "zca_both_separate": top_k(zca_test, zca_items, MAX_K),
        "center_profile": top_k(center_test, items, MAX_K),
        **{
            c: np.tile([item_idx[a] for a in const_lists[c]], (len(test_users), 1))
            for c in ("popularity", "top_category")
        },
    }
    exact_gt = build_ground_truth(cfg, "exact")
    check = {}
    for c, top in ranks.items():
        mine = np.array(
            [_row([idx_to_item[i] for i in t], exact_gt[u])[0] for u, t in zip(test_users, top)]
        )
        check[c] = float(np.abs(mine - ndcg.loc[test_users, c].to_numpy()).max())
    assert max(check.values()) < 1e-4, check
    log.info(f"rankings reproduce the per-user CSV: max |diff| nDCG@10 {check}")

    # B
    rng = np.random.default_rng(SEED)
    sel = np.sort(rng.choice(len(test_users), min(len(test_users), MAX_USERS), replace=False))
    a = rng.integers(0, len(sel), N_PAIRS)
    pairs = np.stack([a, (a + rng.integers(1, len(sel), N_PAIRS)) % len(sel)], axis=1)
    pop = train_popularity(cfg, train_users)
    pop_pct = 100.0 * rankdata([pop.get(idx_to_item[i], 0) for i in range(n_items)]) / n_items
    B = {
        "n_users": len(sel),
        "n_pairs": N_PAIRS,
        "conditions": {
            c: diversity(ranks[c][sel, :TOP], n_items, pop_pct, pairs) for c in B_CONDITIONS
        },
    }
    log.info("B done")

    # E: category concentration and similarity to the constant top-category list
    genre = item_genres(cfg, CAT_DEPTH)
    cat_code = {c: i for i, c in enumerate(sorted(set(genre.values())))}
    item_cat = np.array([cat_code.get(genre.get(idx_to_item[i]), -1) for i in range(n_items)])
    counts = train_category_counts(cfg, train_users, CAT_DEPTH)
    train_order = sorted(counts, key=counts.get, reverse=True)  # ties: first seen, as max()
    assert train_order[0] == max(counts, key=counts.get)
    train_top = [cat_code[c] for c in train_order[:5]]
    users_sel = [test_users[i] for i in sel]
    tc = ranks["top_category"][0, :TOP]
    E = {
        "n_users": len(sel),
        "depth": CAT_DEPTH,
        "n_categories_catalog": len(cat_code),
        "train_top5": [[c, counts[c]] for c in train_order[:5]],
        "concentration": {},
        "jaccard_to_top_category": {},
    }
    jac_all, d2_all = {}, {}
    for c in E_CONDITIONS:
        top = ranks[c][sel, :TOP]
        E["concentration"][c] = category_concentration(
            item_cat[top].ravel(), train_top, len(cat_code)
        )
        jac_all[c] = jaccard_to_list(top, tc)
        d2_all[c] = ndcg_d2.loc[users_sel, c].to_numpy()
        E["jaccard_to_top_category"][c] = {
            "mean_jaccard": float(jac_all[c].mean()),
            "mean_d2_ndcg10": float(d2_all[c].mean()),
            "spearman_vs_d2_ndcg10": spearman(jac_all[c], d2_all[c]),
        }
    for name, conds in (("pooled_personalised", E_PERSONALISED), ("pooled_all", E_CONDITIONS)):
        E["jaccard_to_top_category"][name] = {
            "n_pairs": len(sel) * len(conds),
            "spearman_vs_d2_ndcg10": spearman(
                np.concatenate([jac_all[c] for c in conds]),
                np.concatenate([d2_all[c] for c in conds]),
            ),
        }
    del ranks
    log.info("E done")

    # C: items with a depth-2 category, transforms fitted on the full catalog
    n_levels = category_levels(cfg, set(item_idx))
    genre = item_genres(cfg, CAT_DEPTH)
    rows = np.array([i for i in range(n_items) if n_levels.get(idx_to_item[i], 0) > CAT_DEPTH])
    cats, codes = np.unique([genre[idx_to_item[i]] for i in rows], return_inverse=True)
    sample = rng.choice(len(rows), min(N_NN_ITEMS, len(rows)), replace=False)
    sub = items[rows]
    spaces = {
        "raw": lambda: sub,
        "zca_catalog_a1.0": lambda: zca_items[rows],
        "center_catalog": lambda: center(sub, mom_c.mean),
        "abtt_catalog_D1": lambda: abtt(sub, mom_c.mean, mom_c.eigvecs, 1),
    }
    C = {"n_items": len(rows), "n_categories": len(cats), "n_sample": len(sample), "conditions": {}}
    for c, fn in spaces.items():
        x = fn()
        C["conditions"][c] = {
            "between_within_ratio": between_within(x, codes),
            "nn10_same_category": neighbour_same_category(x, codes, sample),
        }
    log.info("C done")

    # D
    mu = train_prof.mean(0)
    typ = test_prof @ (mu / np.linalg.norm(mu))
    both = ndcg.loc[test_users, ["zca_profile", "raw"]].to_numpy()
    gain = both[:, 0] - both[:, 1]
    nz = (both > 0).any(1)
    rho, p = spearmanr(typ, gain)
    rho_nz, p_nz = spearmanr(typ[nz], gain[nz])
    D = {
        "n_users": len(gain),
        "spearman_rho": float(rho),
        "p": float(p),
        "n_users_nonzero_ndcg": int(nz.sum()),
        "spearman_rho_nonzero": float(rho_nz),
        "p_nonzero": float(p_nz),
        "mean_gain": float(gain.mean()),
    }
    return {"A": A, "B": B, "C": C, "D": D, "E": E, "check_max_abs_diff_ndcg10": check}


if __name__ == "__main__":
    main()
