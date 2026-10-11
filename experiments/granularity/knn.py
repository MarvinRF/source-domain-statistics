"""
Profile-kNN collaborative baseline: score target items with the target
interactions of the TRAIN users whose book profile is most similar.

For a user, the k most similar TRAIN users are found by cosine of their H2
profiles (``raw``: the profiles as they are; ``center_train_users``: both sides
centred with the mean train profile, then L2-normalised). The score of a target
item is the sum, over those neighbours that interacted with it (exact ground
truth), of the cosine similarity. Items nobody in the neighbourhood interacted
with score zero; ties are broken by ascending item id. k in {10, 50, 100, 500, 1000, 2000} is
chosen per profile variant on the VAL users by mean exact nDCG@10 (the smallest k
on ties). Neighbours are always train users; val and test users never are.

kNN + content hybrid: ``cosine_C(profile, item) + lambda * knn(item) / max_i knn(i)``
for the content condition C in {raw, zca_train_users_a1.0}, the neighbourhood of
each profile variant at its VAL-selected k, and the lambda grid of hybrid.py
(0 = content, inf = kNN with ties broken by content); lambda is chosen per
(variant, C) on the VAL users by mean exact nDCG@10 (ties: the smaller lambda).
The pure content conditions (lambda = 0) are checked against the per-user output of
run_corrections (1e-4).

Outputs (results/granularity/; a text other than titlecat adds the suffix _<text>):
  knn_{domain}_{encoder}.csv.gz  per-user TEST metrics of the baselines, the content
                                 conditions, kNN at the selected k and the selected
                                 hybrids (git-ignored)
  knn_{domain}_{encoder}.json    k grid with VAL and TEST curves (selected on VAL),
                                 lambda grid and curves, selections, VAL and TEST means,
                                 grid check, provenance, runtime and peak RSS
  stats_knn.csv   (--stats)  paired tests on TEST; Holm per (domain, metric, family)
  summary_knn.csv (--stats)  TEST means of every condition and domain

Usage:
    python -m experiments.granularity.knn --domain music --encoder gte
    python -m experiments.granularity.knn --domain music --encoder gte --text title
    python -m experiments.granularity.knn --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp

from acw.corrections import center, zca_power
from acw.evaluation.pipeline import MAX_K
from acw.evaluation.ranking import top_k_scores, top_k_scores_ties
from acw.provenance import provenance
from experiments.granularity import common as C
from experiments.granularity.hybrid import LAMBDAS, lam_label
from experiments.granularity.run_corrections import REGISTRY

log = logging.getLogger("knn")

KS = (10, 50, 100, 500, 1000, 2000)
VARIANTS = ("raw", "center_train_users")
CONTENT = ("raw", "zca_train_users_a1.0")
BATCH = 512  # users per score matrix (as ranking.top_k, so the lambda = 0 lists match the grid)
GRID_EQUIV = {"raw": "raw", "zca_train_users_a1.0": "zca_profile"}


def interaction_matrix(pop: C.Population, users: list[str]) -> sp.csr_matrix:
    """Binary (users x catalog) matrix of the exact target interactions of ``users``."""
    indptr, indices = [0], []
    for u in users:
        cols = sorted({pop.item_idx[a] for a in pop.raw_gt.get(u, []) if a in pop.item_idx})
        indices += cols
        indptr.append(len(indices))
    return sp.csr_matrix(
        (np.ones(len(indices), dtype=np.float32), indices, indptr),
        shape=(len(users), len(pop.item_idx)),
    )


def neighbours(sim: np.ndarray, kmax: int) -> tuple[np.ndarray, np.ndarray]:
    """Indices and similarities of the ``kmax`` most similar rows per query, most similar first."""
    idx = top_k_scores(sim, kmax)
    return idx, np.take_along_axis(sim, idx, axis=1)


def knn_scores(idx: np.ndarray, w: np.ndarray, k: int, inter: sp.csr_matrix) -> np.ndarray:
    """Dense (queries x catalog) similarity-weighted neighbour interaction counts, k neighbours."""
    b = idx.shape[0]
    weights = sp.csr_matrix(
        (w[:, :k].ravel(), idx[:, :k].ravel(), np.arange(0, b * k + 1, k)),
        shape=(b, inter.shape[0]),
    )
    return (weights @ inter).toarray()


def normalise(s: np.ndarray) -> np.ndarray:
    """Divide every row by its maximum (rows that are all zero stay zero)."""
    peak = s.max(axis=1, keepdims=True)
    return np.divide(s, peak, out=np.zeros_like(s), where=peak > 0)


def run(domain: str, encoder: str, text: str = "titlecat") -> None:
    t0 = time.time()
    src = REGISTRY[(domain, encoder, text)]
    pop = C.build_population(domain, src)
    emb = C.load_embeddings(src, pop, sources=("train_users",), shrinkage=False)
    mom = emb.moments["train_users"]
    inter = interaction_matrix(pop, pop.train_kept)
    n_users = {s: len(u) for s, u in pop.eval_users.items()}
    profiles = {  # variant -> (train profiles, {split: eval profiles})
        "raw": (emb.train_prof, emb.eval_prof),
        "center_train_users": (
            center(emb.train_prof, mom.mean),
            {s: center(q, mom.mean) for s, q in emb.eval_prof.items()},
        ),
    }
    content_q = {
        "raw": emb.eval_prof,
        "zca_train_users_a1.0": {s: zca_power(q, *mom, 1.0) for s, q in emb.eval_prof.items()},
    }
    n_negative = 0

    def batches(split: str):
        for i in range(0, n_users[split], BATCH):
            yield i, slice(i, i + BATCH)

    # 1. pure kNN for every k: VAL selects k per variant
    pure: dict[str, dict[str, list[np.ndarray]]] = {s: {} for s in pop.eval_users}
    for s in pop.eval_users:
        for variant, (train_p, eval_p) in profiles.items():
            tops = {k: [] for k in KS}
            for _, sl in batches(s):
                idx, w = neighbours(eval_p[s][sl] @ train_p.T, max(KS))
                n_negative += int((w < 0).sum())
                for k in KS:
                    tops[k].append(
                        top_k_scores_ties(knn_scores(idx, w, k, inter), MAX_K, pop.tie_rank)
                    )
            for k in KS:
                pure[s][f"knn_{variant}_k{k}"] = np.vstack(tops[k])
        log.info(f"pure kNN ranked ({s})")
    val_pure = C.score(pop, "val", pure["val"])
    k_star = {
        v: int(C.select_best(val_pure, [f"knn_{v}_k{k}" for k in KS]).rsplit("_k", 1)[1])
        for v in VARIANTS
    }
    log.info(f"selected k on VAL: {k_star}")

    # 2. hybrids at the selected k, every lambda
    combos = [(v, c) for v in VARIANTS for c in CONTENT]
    hyb: dict[str, dict[str, list[np.ndarray]]] = {s: {} for s in pop.eval_users}
    for s in pop.eval_users:
        acc: dict[str, list[np.ndarray]] = {}
        for _, sl in batches(s):
            cos = {c: content_q[c][s][sl] @ emb.items.T for c in CONTENT}
            for variant in VARIANTS:
                train_p, eval_p = profiles[variant]
                idx, w = neighbours(eval_p[s][sl] @ train_p.T, k_star[variant])
                aux = normalise(knn_scores(idx, w, k_star[variant], inter))
                for c in CONTENT:
                    tops = C.lambda_tops(cos[c], aux, pop.tie_rank, MAX_K, plain_at_zero=True)
                    for lam, top in tops.items():
                        acc.setdefault(f"knn_{variant}+{c}|{lam_label(lam)}", []).append(top)
        hyb[s] = {name: np.vstack(v) for name, v in acc.items()}
        log.info(f"hybrids ranked ({s})")

    val = C.score(pop, "val", {**pure["val"], **hyb["val"]})
    test_curve = C.score(pop, "test", {**pure["test"], **hyb["test"]})
    chosen = {f"knn_{v}+{c}": C.select_lambda(val, f"knn_{v}+{c}") for v, c in combos}
    log.info(f"selected lambda: { {b: lam_label(x) for b, x in chosen.items()} }")

    keep: dict[str, np.ndarray] = {}
    rename: dict[str, str] = {}
    for v in VARIANTS:
        name = f"knn_{v}_k{k_star[v]}"
        keep[name] = pure["test"][name]
        rename[name] = name
    for c in CONTENT:  # pure content conditions: lambda = 0, identical for both variants
        keep[c] = hyb["test"][f"knn_{VARIANTS[0]}+{c}|0"]
        rename[f"knn_{VARIANTS[0]}+{c}|0"] = c
    for v, c in combos:
        base = f"knn_{v}+{c}"
        keep[f"hybrid_{base}"] = hyb["test"][f"{base}|{lam_label(chosen[base])}"]
        rename[f"{base}|{lam_label(chosen[base])}"] = f"hybrid_{base}"
    test = C.score(pop, "test", keep, d2=True, strata=True)
    val_sel = val[~val["condition"].str.contains(r"\||_k") | val["condition"].isin(rename)].copy()
    val_sel["condition"] = val_sel["condition"].replace(rename)
    stem_grid = f"{domain}_{encoder}_{text}"
    grid_check = {
        **C.check_against_grid(val_sel, stem_grid, "val", GRID_EQUIV),
        **C.check_against_grid(test, stem_grid, "test", GRID_EQUIV),
    }

    def curve(df: pd.DataFrame, names: list[str]) -> dict:
        m = C.block_means(df)["exact"]
        return {n: m[n] for n in names}

    summary = {
        "domain": domain,
        "encoder": encoder,
        "model": src.model,
        "text": text,
        "score": "sum over the k most similar train users (cosine of H2 profiles) of "
        "similarity * 1[user interacted with the item]",
        "variants": list(VARIANTS),
        "k_grid": list(KS),
        "content": list(CONTENT),
        "selection": "VAL users, mean exact nDCG@10, ties to the smaller k / lambda",
        "selected_k": k_star,
        "selected_lambda": {b: lam_label(x) for b, x in chosen.items()},
        "n_negative_neighbour_weights": n_negative,
        "n_eval_users": n_users,
        "n_train_profiles": len(pop.train_kept),
        "val_k_curve": curve(val_pure, [f"knn_{v}_k{k}" for v in VARIANTS for k in KS]),
        "test_k_curve_diagnostic": curve(
            test_curve, [f"knn_{v}_k{k}" for v in VARIANTS for k in KS]
        ),
        "val_lambda_curve": curve(
            val, [f"knn_{v}+{c}|{lam_label(lam)}" for v, c in combos for lam in LAMBDAS]
        ),
        "test_lambda_curve_diagnostic": curve(
            test_curve, [f"knn_{v}+{c}|{lam_label(lam)}" for v, c in combos for lam in LAMBDAS]
        ),
        "val": C.block_means(val_sel),
        "test": C.block_means(test),
        "grid_check_max_abs_diff": grid_check,
        "provenance": C.provenance_of(pop, *emb.files),
        **C.run_info(t0),
    }
    C.write_run(f"knn_{domain}_{encoder}{C.text_tag(text)}", test, summary)


# ─────────────────────────────────────────────────────────────────────────────
# Paired statistics
# ─────────────────────────────────────────────────────────────────────────────

QN_EXTRA = "center_train_users"  # the other conditions of query_norm are the pure content ones
HY_REFS = ("hybrid_raw", "hybrid_zca_profile")  # hybrid (gte): content + popularity


def stats(encoder: str = "gte", pairset: str = "books") -> None:
    ptag, domains = C.PAIRSETS[pairset]
    rows, summ_rows, files = [], [], []
    for domain, text in C.gte_runs(domains):
        tag = C.text_tag(text)
        stem = f"knn_{domain}_{encoder}{tag}"
        path = C.OUT_DIR / f"{stem}.json"
        if not path.exists():
            log.warning(f"{stem}: no result, skipped")
            continue
        summ = json.loads(path.read_text())
        summ_rows += C.flat_rows(summ)
        k = summ["selected_k"]
        # the popularity hybrids exist for the title+categories text only
        hy_refs = HY_REFS if text == "titlecat" and pairset == "books" else ()
        qn_stem, hy_stem = f"query_norm_{domain}_{encoder}{tag}", f"hybrid_{domain}_{encoder}"
        used = [stem, qn_stem, *([hy_stem] if hy_refs else [])]
        files += [C.OUT_DIR / f"{s}.csv.gz" for s in used]
        wides = {s: C.wide_test(s) for s in used}
        for metric in wides[stem]:
            parts = [wides[stem][metric], wides[qn_stem][metric][[QN_EXTRA]]]
            if hy_refs:
                parts.append(wides[hy_stem][metric][list(hy_refs)])
            assert all(p.index.equals(parts[0].index) for p in parts)
            wide = pd.concat(parts, axis=1)
            meta = {"domain": domain, "encoder": encoder, "text": text, "metric": metric}
            pure = [f"knn_{v}_k{k[v]}" for v in VARIANTS]
            refs = ("popularity", *CONTENT, QN_EXTRA)
            t_pure = [(f"{p}_vs_{r}", p, r) for p in pure for r in refs]
            t_hyb = []
            for v in VARIANTS:
                for c in CONTENT:
                    h = f"hybrid_knn_{v}+{c}"
                    t_hyb += [
                        (f"{h}_vs_popularity", h, "popularity"),
                        (f"{h}_vs_{pure[VARIANTS.index(v)]}", h, pure[VARIANTS.index(v)]),
                        (f"{h}_vs_{c}", h, c),
                        *[(f"{h}_vs_{r}", h, r) for r in hy_refs],
                    ]
            rows += C.paired_rows(wide, t_pure, meta, "knn_vs_references")
            rows += C.paired_rows(wide, t_hyb, meta, "knn_hybrid")
        log.info(f"{stem}: tested")
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / f"stats_knn{ptag}.csv", index=False, float_format="%.6g")
    pd.DataFrame(summ_rows).to_csv(C.OUT_DIR / f"summary_knn{ptag}.csv", index=False, float_format="%.6g")
    prov = {"provenance": provenance(files)}
    (C.OUT_DIR / f"stats_knn_provenance{ptag}.json").write_text(json.dumps(prov, indent=2))
    log.info(f"saved stats_knn.csv ({len(df)} rows) and summary_knn.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=C.DOMAINS + C.DOMAINS2)
    ap.add_argument("--encoder", default="gte")
    ap.add_argument("--text", choices=C.ITEM_TEXTS, default="titlecat")
    ap.add_argument("--pairset", choices=list(C.PAIRSETS), default="books")
    ap.add_argument("--stats", action="store_true", help="paired tests over the finished runs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.stats:
        stats(args.encoder, args.pairset)
    elif args.domain:
        run(args.domain, args.encoder, args.text)
    else:
        ap.error("give --domain or --stats")


if __name__ == "__main__":
    main()
