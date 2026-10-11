"""
Query-normalisation family: which statistics, from which source, applied to the
PROFILE (the query), recover the gain of profile-side ZCA?

For one (domain, encoder, text) (default: the title+categories text), the H2 profile of every
user is transformed with the moments of a source S fitted on

  train_users     the H2 profiles of the TRAIN users
  books_catalog   the embeddings of the source (book) catalog
  target_catalog  the embeddings of the target catalog

and ranked by cosine against the RAW target catalog. Arms per source:

  center_S            subtract the mean of S
  abtt_S_D{1,3,10}    All-but-the-Top with D directions of S
  zca_S_a{0.25,0.5,0.75,1.0}   partial ZCA with the covariance of S
  zcalw_S             ZCA (alpha 1) with the Ledoit-Wolf shrunk covariance of S
                      (sklearn's intensity, eigenvalues shrunk towards their mean)

Reference and catalog-side arms (profile raw unless stated):

  raw                         raw / raw
  center_catalog, abtt_catalog_D{1,3,10}, zca_catalog_a{0.25,0.5,0.75,1.0},
  zca_catalog_books           the catalog-side arms of run_corrections (items corrected
                              with the catalog's own, or the book catalog's, statistics)
  zca_catalog_train_users     raw / ZCA of the catalog with the train-profile covariance
  zca_both_separate           zca_train_users_a1.0 / zca_catalog_a1.0

``zca_train_users_a1.0`` is run_corrections' ``zca_profile``; every arm that
run_corrections also computes and the baselines are checked against its per-user
output (1e-4). VAL users (exact nDCG@10) select the best arm of each source, the best
query-side arm and the best catalog-side arm (``pca_catalog`` and the off-domain
grocery control are not part of the family); TEST metrics are reported for all arms
under the exact ground truth, depth-2 categories and the cold / warm subsets of the
exact ground truth.

Outputs (results/granularity/; a text other than titlecat adds the suffix _<text>):
  query_norm_{domain}_{encoder}.csv.gz  per-user TEST metrics (git-ignored)
  query_norm_{domain}_{encoder}.json    VAL and TEST means, selections, Ledoit-Wolf
                                        intensities, grid check, provenance, runtime
  stats_query_norm.csv   (--stats)  paired tests on TEST: the contrasts fixed a priori
                          (a: query-side vs catalog-side, b: ZCA vs centering with the
                          same statistics, c: both sides vs profile only), the contrast
                          of the VAL-selected arms, and arms vs raw; Holm per family
                          Three post hoc contrasts (family ``post_hoc``, Holm among the three): the
                          query ZCA / centring with TARGET-catalog statistics against raw, and the
                          users' ZCA against the target-catalog ZCA on the query
                          A fourth post hoc family ``matrix`` (11 tests, Holm among them) covers the
                          side x source matrix of ZCA: the six cells against raw, the three
                          matched-source side contrasts (query vs items with the same source) and
                          the source contrasts (U vs target catalog, B vs target catalog; query, items)
  tally_query_norm.csv   (--stats)  better / worse / ns counts of those tests
  summary_query_norm.csv (--stats)  TEST means of every condition of every run
  query_norm_{domain}_{encoder}_val.csv.gz  per-user VAL metrics (exact level; git-ignored)
  stats_query_norm_val.csv (--stats-val)  the contrasts, arms against raw, post hoc and matrix
                          tests again on the VAL users (replication; the contrasts that use the
                          VAL-selected arms are left out because the arms were chosen on those users)

Usage:
    python -m experiments.granularity.query_norm --domain music --encoder gte
    python -m experiments.granularity.query_norm --domain music --encoder gte --text title
    python -m experiments.granularity.query_norm --stats
    python -m experiments.granularity.query_norm --stats-val
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Callable
from functools import partial

import numpy as np
import pandas as pd

from acw.corrections import Moments, abtt, center, zca_power
from acw.evaluation.pipeline import MAX_K
from acw.evaluation.ranking import top_k
from acw.provenance import provenance
from experiments.granularity import common as C
from experiments.granularity.run_corrections import REGISTRY, UNVERIFIED_ENCODERS

log = logging.getLogger("query_norm")

ALPHAS = (0.25, 0.5, 0.75, 1.0)
ABTT_D = (1, 3, 10)
CATALOG_ARMS = (
    "center_catalog",
    *(f"abtt_catalog_D{D}" for D in ABTT_D),
    *(f"zca_catalog_a{a}" for a in ALPHAS),
    "zca_catalog_books",
    "zca_catalog_train_users",
)
GRID_EQUIV = {  # condition -> condition of run_corrections
    "raw": "raw",
    "zca_train_users_a1.0": "zca_profile",
    "zca_both_separate": "zca_both_separate",
    **{a: a for a in CATALOG_ARMS if a != "zca_catalog_train_users"},
}
# Arms and contrasts fixed before looking at any test number
A_PRIORI = (
    "center_train_users",
    "zca_train_users_a1.0",
    "zca_books_catalog_a1.0",
    "zca_catalog_a1.0",
    "zca_both_separate",
)
Q = Callable[[np.ndarray], np.ndarray]


def query_arms(moments: dict[str, Moments], shrunk: dict[str, Moments]) -> dict[str, tuple[str, Q]]:
    """Name -> (statistics source, query transform) of every query-side arm, in a fixed order."""
    arms: dict[str, tuple[str, Q]] = {}
    for s in C.SOURCES:
        m = moments[s]
        arms[f"center_{s}"] = (s, partial(center, mean=m.mean))
        for D in ABTT_D:
            arms[f"abtt_{s}_D{D}"] = (s, partial(abtt, mean=m.mean, components=m.eigvecs, D=D))
        for a in ALPHAS:
            arms[f"zca_{s}_a{a}"] = (s, partial(zca_power, mean=m.mean, eigvals=m.eigvals,
                                                eigvecs=m.eigvecs, alpha=a))  # fmt: skip
        w = shrunk[s]
        arms[f"zcalw_{s}"] = (s, partial(zca_power, mean=w.mean, eigvals=w.eigvals,
                                         eigvecs=w.eigvecs, alpha=1.0))  # fmt: skip
    return arms


def run(domain: str, encoder: str, text: str = "titlecat") -> None:
    t0 = time.time()
    src = REGISTRY[(domain, encoder, text)]
    pop = C.build_population(domain, src)
    emb = C.load_embeddings(src, pop)
    items = emb.items
    arms = query_arms(emb.moments, emb.shrunk)
    mom_c, mom_u, mom_b = (
        emb.moments[s] for s in ("target_catalog", "train_users", "books_catalog")
    )
    zca_c = zca_power(items, *mom_c, 1.0)

    def ident(q: np.ndarray) -> np.ndarray:
        return q

    # name -> (query transform, thunk building the item matrix)
    specs: dict[str, tuple[Q, Callable[[], np.ndarray]]] = {"raw": (ident, lambda: items)}
    specs.update({name: (fn, lambda: items) for name, (_, fn) in arms.items()})
    specs["center_catalog"] = (ident, lambda: center(items, mom_c.mean))
    for D in ABTT_D:
        specs[f"abtt_catalog_D{D}"] = (ident, partial(abtt, items, mom_c.mean, mom_c.eigvecs, D))
    for a in ALPHAS:
        specs[f"zca_catalog_a{a}"] = (
            (ident, lambda: zca_c) if a == 1.0 else (ident, partial(zca_power, items, *mom_c, a))
        )
    specs["zca_catalog_books"] = (ident, partial(zca_power, items, *mom_b, 1.0))
    specs["zca_catalog_train_users"] = (ident, partial(zca_power, items, *mom_u, 1.0))
    specs["zca_both_separate"] = (arms["zca_train_users_a1.0"][1], lambda: zca_c)
    rankings: dict[str, dict[str, np.ndarray]] = {s: {} for s in pop.eval_users}
    for name, (qfn, xfn) in specs.items():
        x = xfn()
        for s, q in emb.eval_prof.items():
            rankings[s][name] = top_k(qfn(q), x, MAX_K)
        del x
        log.info(f"ranked {name}")
    zca_c = None  # release the whitened catalog (the thunks above are no longer called)

    val = C.score(pop, "val", rankings["val"])
    test = C.score(pop, "test", rankings["test"], d2=True, strata=True)
    stem_grid = f"{domain}_{encoder}_{text}"
    grid_check = {
        **C.check_against_grid(val, stem_grid, "val", GRID_EQUIV),
        **C.check_against_grid(test, stem_grid, "test", GRID_EQUIV),
    }
    selected = {
        "per_source": {
            s: C.select_best(val, [n for n, (src_, _) in arms.items() if src_ == s])
            for s in C.SOURCES
        },
        "query_side": C.select_best(val, list(arms)),
        "catalog_side": C.select_best(val, list(CATALOG_ARMS)),
    }
    log.info(f"selected on VAL: {selected}")

    stem = f"query_norm_{domain}_{encoder}{C.text_tag(text)}"
    summary = {
        "domain": domain,
        "encoder": encoder,
        "model": src.model,
        "text": text,
        "verified": encoder not in UNVERIFIED_ENCODERS,
        "n_eval_users": {s: len(u) for s, u in pop.eval_users.items()},
        "n_train_profiles": len(pop.train_kept),
        "query_arms": {n: s for n, (s, _) in arms.items()},
        "a_priori": list(A_PRIORI),
        "catalog_arms": list(CATALOG_ARMS),
        "selection": "VAL users, mean exact nDCG@10, ties to the first arm of the listed order",
        "selected": selected,
        "ledoit_wolf_intensity": emb.shrinkage,
        "val": C.block_means(val),
        "test": C.block_means(test),
        "grid_check_max_abs_diff": grid_check,
        "provenance": C.provenance_of(pop, *emb.files),
        **C.run_info(t0),
    }
    C.write_run(stem, test, summary)
    val.to_csv(C.OUT_DIR / f"{stem}_val.csv.gz", index=False, float_format="%.6g")


# ─────────────────────────────────────────────────────────────────────────────
# Paired statistics over the runs
# ─────────────────────────────────────────────────────────────────────────────


def contrasts(sel: dict) -> list[tuple[str, str, str]]:
    """(test id, condition, reference) of the contrast family of one run."""
    return [
        ("a_zca_train_users_vs_zca_catalog", "zca_train_users_a1.0", "zca_catalog_a1.0"),
        ("a_center_train_users_vs_zca_catalog", "center_train_users", "zca_catalog_a1.0"),
        ("a_zca_books_catalog_vs_zca_catalog", "zca_books_catalog_a1.0", "zca_catalog_a1.0"),
        ("a_selected_query_vs_selected_catalog", sel["query_side"], sel["catalog_side"]),
        ("b_zca_vs_center_train_users", "zca_train_users_a1.0", "center_train_users"),
        ("c_both_sides_vs_query_only", "zca_both_separate", "zca_train_users_a1.0"),
    ]


def versus_raw(sel: dict) -> list[tuple[str, str, str]]:
    """Arms fixed a priori, the extra catalog-side arm and the VAL-selected arms, each vs raw."""
    names = [
        *A_PRIORI,
        "zca_catalog_train_users",
        sel["query_side"],
        sel["catalog_side"],
        *sel["per_source"].values(),
    ]
    return [(f"vs_raw:{n}", n, "raw") for n in dict.fromkeys(names)]


def post_hoc() -> list[tuple[str, str, str]]:
    """Contrasts added after the first results were seen, Holm-adjusted as their own family:
    do the TARGET-catalog statistics applied to the query help, and are the users' better?"""
    return [
        ("post_hoc:zca_target_catalog_vs_raw", "zca_target_catalog_a1.0", "raw"),
        ("post_hoc:center_target_catalog_vs_raw", "center_target_catalog", "raw"),
        (
            "post_hoc:zca_train_users_vs_zca_target_catalog",
            "zca_train_users_a1.0",
            "zca_target_catalog_a1.0",
        ),
    ]


def matrix() -> list[tuple[str, str, str]]:
    """Post hoc family (defined after the third review round): the side x source matrix of ZCA
    (alpha 1). Query cells transform the query, item cells the items; U = train users' profiles,
    B = book catalog, C = target catalog. Six cells against raw, the matched-source side contrasts
    (query minus items, same source) and the contrasts of the sources inside one side."""
    q = {"U": "zca_train_users_a1.0", "B": "zca_books_catalog_a1.0", "C": "zca_target_catalog_a1.0"}
    i = {"U": "zca_catalog_train_users", "B": "zca_catalog_books", "C": "zca_catalog_a1.0"}
    return [
        *[(f"matrix:query_{k}_vs_raw", q[k], "raw") for k in "UBC"],
        *[(f"matrix:items_{k}_vs_raw", i[k], "raw") for k in "UBC"],
        *[(f"matrix:side_{k}", q[k], i[k]) for k in "UBC"],
        ("matrix:source_query_U_vs_C", q["U"], q["C"]),
        ("matrix:source_query_B_vs_C", q["B"], q["C"]),
        ("matrix:source_items_U_vs_C", i["U"], i["C"]),
        ("matrix:source_items_B_vs_C", i["B"], i["C"]),
    ]


def stats(split: str = "test", pairset: str = "books") -> None:
    """Paired tests of the finished runs on the TEST users, or (``split="val"``) the replication
    on the VAL users: the same families without the tests that use the VAL-selected arms."""
    val = split == "val"
    suffix = "_val" if val else ""
    ptag, domains = C.PAIRSETS[pairset]
    rows, summ_rows, files = [], [], []
    for domain, encoder, text in C.verified_runs(domains):
        stem = f"query_norm_{domain}_{encoder}{C.text_tag(text)}"
        path = C.OUT_DIR / f"{stem}.json"
        if not path.exists() or (val and not (C.OUT_DIR / f"{stem}_val.csv.gz").exists()):
            log.warning(f"{stem}: no result, skipped")
            continue
        summ = json.loads(path.read_text())
        files.append(C.OUT_DIR / f"{stem}{suffix}.csv.gz")
        summ_rows += C.flat_rows(summ)
        sel = summ["selected"]
        con = [c for c in contrasts(sel) if not (val and c[0].startswith("a_selected"))]
        vs = versus_raw(sel)
        if val:  # the arms chosen on VAL are not independent of the VAL users
            vs = [t for t in vs if t[1] in (*A_PRIORI, "zca_catalog_train_users")]
        for metric, wide in C.wide_test(f"{stem}{suffix}").items():
            meta = {"domain": domain, "encoder": encoder, "text": text, "metric": metric}
            rows += C.paired_rows(wide, con, meta, "contrast")
            rows += C.paired_rows(wide, vs, meta, "vs_raw")
            rows += C.paired_rows(wide, post_hoc(), meta, "post_hoc")
            rows += C.paired_rows(wide, matrix(), meta, "matrix")
        log.info(f"{stem}{suffix}: tested")
    df = pd.DataFrame(rows)
    if val:
        df.to_csv(C.OUT_DIR / f"stats_query_norm_val{ptag}.csv", index=False, float_format="%.6g")
        prov = {"runs": [p.name for p in files], "provenance": provenance(files)}
        (C.OUT_DIR / f"stats_query_norm_val_provenance{ptag}.json").write_text(json.dumps(prov, indent=2))
        log.info(f"saved stats_query_norm_val{ptag}.csv ({len(df)} rows)")
        return
    df.to_csv(C.OUT_DIR / f"stats_query_norm{ptag}.csv", index=False, float_format="%.6g")
    tl = C.tally(rows, ["domain", "text", "family", "test", "metric"])
    tl.to_csv(C.OUT_DIR / f"tally_query_norm{ptag}.csv", index=False)
    pd.DataFrame(summ_rows).to_csv(C.OUT_DIR / f"summary_query_norm{ptag}.csv", index=False,
                                   float_format="%.6g")  # fmt: skip
    prov = {"runs": [p.name for p in files], "provenance": provenance(files)}
    (C.OUT_DIR / f"stats_query_norm_provenance{ptag}.json").write_text(json.dumps(prov, indent=2))
    log.info(
        f"saved stats_query_norm.csv ({len(df)} rows), tally_query_norm.csv, summary_query_norm.csv"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=C.DOMAINS + C.DOMAINS2)
    ap.add_argument("--encoder", default="gte")
    ap.add_argument("--text", choices=C.ITEM_TEXTS, default="titlecat")
    ap.add_argument("--stats", action="store_true", help="paired tests over the finished runs")
    ap.add_argument("--pairset", choices=list(C.PAIRSETS), default="books")
    ap.add_argument("--stats-val", action="store_true", help="the same tests on the VAL users")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.stats_val:
        stats("val", args.pairset)
    elif args.stats:
        stats(pairset=args.pairset)
    elif args.domain:
        run(args.domain, args.encoder, args.text)
    else:
        ap.error("give --domain or --stats")


if __name__ == "__main__":
    main()
