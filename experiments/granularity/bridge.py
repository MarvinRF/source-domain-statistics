"""
Closed-form ridge bridge (EMCDR-style) from the book profile to the target catalog.

On the TRAIN users, a linear map W (with intercept) sends the H2 book profile x to the mean
embedding y of the user's train target items (exact interactions that are in the catalog; users
without one are left out): ``W = argmin ||Y - XW||^2 + lambda ||W||^2`` on centred X and Y, solved in
closed form from one eigendecomposition of X'X. A user is scored by the cosine between the mapped
profile ``(x - mean_X) W + mean_Y`` and the (L2-normalised) item embeddings. lambda = s * tr(X'X)/d
over the grid ``SCALES`` is chosen on the VAL users by mean exact nDCG@10 (the larger lambda on
ties). Val and test users never enter the fit. The raw and query-ZCA(U) conditions are re-scored
as in knn.py (checked against the grid, 1e-4); the bridge is tested against both.

Outputs (results/granularity/; a text other than titlecat adds the suffix _<text>):
  bridge_{domain}_{encoder}.csv.gz / .json   per-user TEST metrics; lambda curve (VAL, TEST
                                             diagnostic), selection, means, provenance
  stats_bridge.csv, summary_bridge.csv  (--stats)  paired tests on TEST (one Holm family per
                                             domain and metric)

Usage:
    python -m experiments.granularity.bridge --domain music --encoder gte
    python -m experiments.granularity.bridge --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from acw.corrections import zca_power
from acw.evaluation.pipeline import MAX_K
from acw.evaluation.ranking import top_k
from acw.provenance import provenance
from experiments.granularity import common as C
from experiments.granularity.knn import CONTENT, GRID_EQUIV, interaction_matrix
from experiments.granularity.run_corrections import REGISTRY

log = logging.getLogger("bridge")

SCALES = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)  # lambda / (tr(X'X) / d)
BRIDGE = "bridge_ridge"


def fit_ridge_path(X: np.ndarray, Y: np.ndarray, scales=SCALES):
    """Means and the ridge weights of every scale: ``{s: W}``, lambda = s * tr(X'X)/d on the
    centred data, W solved in closed form from the eigendecomposition of X'X."""
    mx, my = X.mean(0), Y.mean(0)
    Xc, Yc = (X - mx).astype(np.float64), (Y - my).astype(np.float64)
    lam, V = np.linalg.eigh(Xc.T @ Xc)
    xy = V.T @ (Xc.T @ Yc)
    unit = lam.sum() / len(lam)
    return mx, my, {s: V @ (xy / (lam + s * unit)[:, None]) for s in scales}


def map_profiles(Q: np.ndarray, mx, my, W) -> np.ndarray:
    return ((Q - mx) @ W + my).astype(np.float32)


def run(domain: str, encoder: str, text: str = "titlecat") -> None:
    t0 = time.time()
    src = REGISTRY[(domain, encoder, text)]
    pop = C.build_population(domain, src)
    emb = C.load_embeddings(src, pop, sources=("train_users",), shrinkage=False)
    mom = emb.moments["train_users"]
    inter = interaction_matrix(pop, pop.train_kept)
    n_i = np.asarray(inter.sum(axis=1)).ravel()
    keep = n_i > 0
    Y = (inter @ emb.items)[keep] / n_i[keep, None]
    X = emb.train_prof[keep]
    mx, my, Ws = fit_ridge_path(X, Y)
    items = emb.items / np.linalg.norm(emb.items, axis=1, keepdims=True)
    log.info(f"{domain}: ridge fitted on {len(X)} of {len(pop.train_kept)} train users")

    def tops(split: str) -> dict[str, np.ndarray]:
        q = emb.eval_prof[split]
        return {f"{BRIDGE}_s{s:g}": top_k(map_profiles(q, mx, my, W), items, MAX_K)
                for s, W in Ws.items()}  # fmt: skip

    val_curve = C.score(pop, "val", tops("val"))
    names = [f"{BRIDGE}_s{s:g}" for s in reversed(SCALES)]  # larger lambda first: wins ties
    best = C.select_best(val_curve, names)
    s_star = float(best.rsplit("_s", 1)[1])
    log.info(f"selected lambda scale on VAL: {s_star:g}")
    test_tops = tops("test")
    content = {
        "raw": top_k(emb.eval_prof["test"], emb.items, MAX_K),
        "zca_train_users_a1.0": top_k(
            zca_power(emb.eval_prof["test"], *mom, 1.0), emb.items, MAX_K
        ),
    }
    test = C.score(pop, "test", {BRIDGE: test_tops[best], **content}, d2=True, strata=True)
    test_curve = C.score(pop, "test", test_tops)
    val_sel = C.score(pop, "val", {
        "raw": top_k(emb.eval_prof["val"], emb.items, MAX_K),
        "zca_train_users_a1.0": top_k(zca_power(emb.eval_prof["val"], *mom, 1.0), emb.items, MAX_K),
    })  # fmt: skip
    stem_grid = f"{domain}_{encoder}_{text}"
    grid_check = {**C.check_against_grid(val_sel, stem_grid, "val", GRID_EQUIV),
                  **C.check_against_grid(test, stem_grid, "test", GRID_EQUIV)}  # fmt: skip

    def curve(df: pd.DataFrame) -> dict:
        m = C.block_means(df)["exact"]
        return {n: m[n] for n in names}

    summary = {
        "domain": domain, "encoder": encoder, "model": src.model, "text": text,
        "model_description": "ridge map of the H2 book profile to the mean embedding of the "
        "user's train target items (centred, with intercept); cosine to the items",
        "scales": list(SCALES), "selected_scale": s_star,
        "selection": "VAL users, mean exact nDCG@10, ties to the larger lambda",
        "n_fit_users": int(keep.sum()), "n_train_profiles": len(pop.train_kept),
        "n_eval_users": {s: len(u) for s, u in pop.eval_users.items()},
        "val_lambda_curve": curve(val_curve), "test_lambda_curve_diagnostic": curve(test_curve),
        "test": C.block_means(test), "grid_check_max_abs_diff": grid_check,
        "provenance": C.provenance_of(pop, *emb.files), **C.run_info(t0),
    }  # fmt: skip
    C.write_run(f"bridge_{domain}_{encoder}{C.text_tag(text)}", test, summary)


def stats(encoder: str = "gte") -> None:
    rows, summ_rows, files = [], [], []
    for domain, text in C.gte_runs():
        stem = f"bridge_{domain}_{encoder}{C.text_tag(text)}"
        path = C.OUT_DIR / f"{stem}.json"
        if not path.exists():
            log.warning(f"{stem}: no result, skipped")
            continue
        summ_rows += C.flat_rows(json.loads(path.read_text()))
        files.append(C.OUT_DIR / f"{stem}.csv.gz")
        for metric, wide in C.wide_test(stem).items():
            meta = {"domain": domain, "encoder": encoder, "text": text, "metric": metric}
            pairs = [(f"{BRIDGE}_vs_{r}", BRIDGE, r) for r in (*CONTENT, "popularity")]
            rows += C.paired_rows(wide, pairs, meta, "bridge_vs_references")
        log.info(f"{stem}: tested")
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "stats_bridge.csv", index=False, float_format="%.6g")
    pd.DataFrame(summ_rows).to_csv(C.OUT_DIR / "summary_bridge.csv", index=False,
                                   float_format="%.6g")  # fmt: skip
    (C.OUT_DIR / "stats_bridge_provenance.json").write_text(
        json.dumps({"provenance": provenance(files)}, indent=2))  # fmt: skip


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=C.DOMAINS)
    ap.add_argument("--encoder", default="gte")
    ap.add_argument("--text", choices=C.ITEM_TEXTS, default="titlecat")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.stats:
        stats(args.encoder)
    elif args.domain:
        run(args.domain, args.encoder, args.text)
    else:
        ap.error("give --domain or --stats")


if __name__ == "__main__":
    main()
