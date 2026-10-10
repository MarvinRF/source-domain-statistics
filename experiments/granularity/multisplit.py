"""
Split-to-split variability of the key contrasts.

The main split (seed 42, results/granularity/) and the additional splits of seeds 43-46
(results/granularity/seed{S}/, produced with ``ACW_SPLIT_SEED=S``; see docs/REPRODUCIBILITY.md)
each have their own train / validation / test users, so every fitted quantity (statistics of the
training users, popularity, neighbourhoods, the validation choices of lambda and k) is refitted.
For the GTE title+categories runs of the three domains this script reads the paired tests of
every split (stats_query_norm.csv, stats_hybrid.csv, stats_knn.csv, stats_lexical.csv,
stats_vs_popularity.csv; exact-level nDCG@10, the Holm-adjusted sign-flip p of the split's own
family) and aggregates, per (contrast, domain):

  mean_diff_mean / mean_diff_sd   mean and sample standard deviation (n - 1) of the paired mean
                                  difference over the splits (first condition minus the second)
  n_better / n_worse / n_ns       splits where the contrast is significantly positive / negative /
                                  not significant (Holm sign-flip p < 0.05)

Outputs (results/granularity/):
  multisplit_per_split.csv  one row per (contrast, domain, seed)
  multisplit_summary.csv    one row per (contrast, domain)
  multisplit_centring_share.csv  per domain: share of the query-ZCA gain recovered by centring
  multisplit_levels.csv     exact nDCG@10 and users with a hit of the main conditions per split
  multisplit_provenance.json  provenance of the files read

Usage:
    python -m experiments.granularity.multisplit
"""

from __future__ import annotations

import json
import logging
import re

import numpy as np
import pandas as pd

from acw.provenance import provenance
from acw.settings import RESULTS_DIR, SEED

log = logging.getLogger("multisplit")

BASE = RESULTS_DIR / "granularity"
SEEDS = (SEED, 43, 44, 45, 46)
METRIC = "ndcg10"
ALPHA = 0.05
DOMAINS = ("music", "movies", "games")


def seed_dir(seed: int):
    return BASE if seed == SEED else BASE / f"seed{seed}"


def _pat(pattern: str):
    rx = re.compile(pattern)
    return lambda s: bool(rx.fullmatch(s))


# (id, label, file, column holding the test name, predicate on that column)
# Labels use the paper's notation: U = training users, B = book catalog, C = target catalog.
# `test` of stats_query_norm / stats_knn / stats_lexical; `cond|ref` for stats_hybrid and
# stats_vs_popularity (built below as "condition_vs_reference").
SPECS = [
    ("qU_vs_raw", "Query ZCA(U) vs raw", "stats_query_norm.csv", "vs_raw:zca_train_users_a1.0"),
    ("qCtr_vs_raw", "Query centring(U) vs raw", "stats_query_norm.csv", "vs_raw:center_train_users"),
    ("a1", "Query ZCA(U) vs items ZCA(C) (a1)", "stats_query_norm.csv", "a_zca_train_users_vs_zca_catalog"),
    ("a2", "Query centring(U) vs items ZCA(C) (a2)", "stats_query_norm.csv", "a_center_train_users_vs_zca_catalog"),
    ("a3", "Query ZCA(B) vs items ZCA(C) (a3)", "stats_query_norm.csv", "a_zca_books_catalog_vs_zca_catalog"),
    ("a4", "Best query arm vs best items arm (a4)", "stats_query_norm.csv", "a_selected_query_vs_selected_catalog"),
    ("b", "Query ZCA(U) vs centring(U) (b)", "stats_query_norm.csv", "b_zca_vs_center_train_users"),
    ("c", "Both sides vs query only (c)", "stats_query_norm.csv", "c_both_sides_vs_query_only"),
    ("d1", "Query ZCA(C) vs raw (d1)", "stats_query_norm.csv", "post_hoc:zca_target_catalog_vs_raw"),
    ("d2", "Query ZCA(U) vs query ZCA(C) (d2)", "stats_query_norm.csv", "post_hoc:zca_train_users_vs_zca_target_catalog"),
    ("itemsU_vs_raw", "Items ZCA(U) vs raw", "stats_query_norm.csv", "matrix:items_U_vs_raw"),
    ("itemsB_vs_raw", "Items ZCA(B) vs raw", "stats_query_norm.csv", "matrix:items_B_vs_raw"),
    ("itemsC_vs_raw", "Items ZCA(C) vs raw", "stats_query_norm.csv", "matrix:items_C_vs_raw"),
    ("sideU", "Query vs items, same source U", "stats_query_norm.csv", "matrix:side_U"),
    ("sideB", "Query vs items, same source B", "stats_query_norm.csv", "matrix:side_B"),
    ("sideC", "Query vs items, same source C", "stats_query_norm.csv", "matrix:side_C"),
    ("lexIdf_vs_ctr", "IDF overlap vs centred query", "stats_lexical.csv", "idf_overlap_title_vs_center_train_users"),
    ("lexBm25_vs_ctr", "BM25 vs centred query", "stats_lexical.csv", "bm25_title_vs_center_train_users"),
    ("lexIdf_vs_zca", "IDF overlap vs query ZCA(U)", "stats_lexical.csv", "idf_overlap_title_vs_zca_train_users_a1.0"),
    ("lexBm25_vs_zca", "BM25 vs query ZCA(U)", "stats_lexical.csv", "bm25_title_vs_zca_train_users_a1.0"),
    ("zca_vs_pop", "Query ZCA(U) vs popularity", "stats_vs_popularity.csv", "zca_profile"),
    ("hybZca_vs_pop", "Hybrid ZCA + popularity vs popularity", "stats_hybrid.csv", "hybrid_zca_profile_vs_popularity"),
    ("hybZca_vs_hybRaw", "Hybrid ZCA + popularity vs hybrid raw", "stats_hybrid.csv", "hybrid_zca_profile_vs_hybrid_raw"),
    ("knn_vs_zca", "Centred kNN vs query ZCA(U)", "stats_knn.csv", r"knn_center_train_users_k\d+_vs_zca_train_users_a1\.0"),
    ("knn_vs_pop", "Centred kNN vs popularity", "stats_knn.csv", r"knn_center_train_users_k\d+_vs_popularity"),
    ("knnHyb_vs_knn", "kNN + ZCA vs centred kNN", "stats_knn.csv", r"hybrid_knn_center_train_users\+zca_train_users_a1\.0_vs_knn_center_train_users_k\d+"),
    ("knnHyb_vs_pop", "kNN + ZCA vs popularity", "stats_knn.csv", r"hybrid_knn_center_train_users\+zca_train_users_a1\.0_vs_popularity"),
]  # fmt: skip
SPEC_IDS = [s[0] for s in SPECS]


def _load(seed: int, name: str) -> pd.DataFrame | None:
    path = seed_dir(seed) / name
    if not path.exists():
        return None
    df = pd.read_csv(path)
    df = df[df["metric"] == METRIC]
    if "encoder" in df:
        df = df[df["encoder"] == "gte"]
    if "text" in df:
        df = df[df["text"] == "titlecat"]
    if "gt" in df:
        df = df[df["gt"] == "exact"]
    return df


def _test_names(df: pd.DataFrame, file: str) -> pd.Series:
    if file == "stats_hybrid.csv":
        return df["condition"] + "_vs_" + df["reference"]
    if file == "stats_vs_popularity.csv":
        return df["condition"]
    return df["test"]


def verdict(p: float, diff: float) -> str:
    return "better" if p < ALPHA and diff > 0 else "worse" if p < ALPHA and diff < 0 else "ns"


def per_split() -> tuple[pd.DataFrame, list]:
    rows, files = [], []
    for seed in SEEDS:
        cache: dict[str, pd.DataFrame | None] = {}
        for cid, label, file, pattern in SPECS:
            if file not in cache:
                cache[file] = _load(seed, file)
                if cache[file] is not None:
                    files.append(seed_dir(seed) / file)
            df = cache[file]
            if df is None:
                continue
            match = _pat(pattern)
            sel = df[_test_names(df, file).map(match)]
            if (
                file == "stats_query_norm.csv"
            ):  # a test name occurs once per (domain, text, encoder)
                assert not sel.duplicated("domain").any(), (seed, cid)
            for _, r in sel.iterrows():
                rows.append(
                    {
                        "contrast": cid,
                        "label": label,
                        "domain": r["domain"],
                        "seed": seed,
                        "n": int(r["n"]),
                        "mean_diff": r["mean_diff"],
                        "ci95_low": r["ci95_low"],
                        "ci95_high": r["ci95_high"],
                        "n_nonzero_diff": int(r["n_nonzero_diff"]),
                        "p_holm_signflip": r["p_holm_signflip"],
                        "verdict": verdict(r["p_holm_signflip"], r["mean_diff"]),
                    }
                )
    df = pd.DataFrame(rows)
    df["contrast"] = pd.Categorical(df["contrast"], SPEC_IDS, ordered=True)
    return df.sort_values(["contrast", "domain", "seed"]).reset_index(drop=True), files


def summarise(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["contrast", "domain"], observed=True)
    out = g.agg(
        label=("label", "first"),
        n_splits=("seed", "nunique"),
        mean_diff_mean=("mean_diff", "mean"),
        mean_diff_sd=("mean_diff", lambda x: x.std(ddof=1) if len(x) > 1 else np.nan),
        mean_diff_min=("mean_diff", "min"),
        mean_diff_max=("mean_diff", "max"),
        n_better=("verdict", lambda v: int((v == "better").sum())),
        n_worse=("verdict", lambda v: int((v == "worse").sum())),
        n_ns=("verdict", lambda v: int((v == "ns").sum())),
        n_sign_positive=("mean_diff", lambda x: int((x > 0).sum())),
        n_nonzero_mean=("n_nonzero_diff", "mean"),
    ).reset_index()
    return out


def centring_share(df: pd.DataFrame) -> pd.DataFrame:
    """Per split, the share of the query-ZCA gain over raw that query centring recovers,
    qCtr_vs_raw / qU_vs_raw (mean differences); mean, sd (n - 1), min, max over the splits."""
    w = df[df["contrast"].isin(["qCtr_vs_raw", "qU_vs_raw"])].pivot_table(
        index=["domain", "seed"], columns="contrast", values="mean_diff", observed=True
    )
    w["share"] = w["qCtr_vs_raw"] / w["qU_vs_raw"]
    g = w.groupby("domain")["share"]
    return g.agg(n_splits="count", share_mean="mean", share_sd="std", share_min="min", share_max="max").reset_index()  # fmt: skip


def levels() -> pd.DataFrame:
    """Exact nDCG@10 and users with a top-10 hit of the main conditions, per split and domain."""
    keep = ["raw", "center_train_users", "zca_train_users_a1.0", "zca_catalog_a1.0",
            "zca_catalog_train_users", "popularity", "top_category", "random"]  # fmt: skip
    rows = []
    for seed in SEEDS:
        path = seed_dir(seed) / "summary_query_norm.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df = df[(df["encoder"] == "gte") & (df["text"] == "titlecat") & df["condition"].isin(keep)]
        for _, r in df.iterrows():
            rows.append(
                {"seed": seed, "domain": r["domain"], "condition": r["condition"],
                 "ndcg10": r["ndcg10"], "n_users": int(r["n_users"]), "n_hit10": int(r["n_hit10"])}
            )  # fmt: skip
    return pd.DataFrame(rows)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    df, files = per_split()
    summary = summarise(df)
    lv = levels()
    centring_share(df).to_csv(BASE / "multisplit_centring_share.csv", index=False, float_format="%.6g")
    df.to_csv(BASE / "multisplit_per_split.csv", index=False, float_format="%.6g")
    summary.to_csv(BASE / "multisplit_summary.csv", index=False, float_format="%.6g")
    lv.to_csv(BASE / "multisplit_levels.csv", index=False, float_format="%.6g")
    prov = {"seeds": list(SEEDS), "domains": list(DOMAINS), "provenance": provenance(files)}
    (BASE / "multisplit_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info(
        f"saved multisplit_per_split.csv ({len(df)} rows), multisplit_summary.csv "
        f"({len(summary)} rows), multisplit_levels.csv"
    )


if __name__ == "__main__":
    main()
