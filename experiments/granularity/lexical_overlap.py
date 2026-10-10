"""
Is the item-level signal of the normalised query lexical?  Hit overlap and semantic residual.

For one domain (GTE, title+categories embeddings, TEST users) the script ranks the target catalog
with the raw, the centred and the ZCA query (train-user statistics: the arms of query_norm.py)
and with three embedding-free scorers (IDF overlap and BM25 over the item titles, and the
stricter IDF-overlap matcher of lexical.py), and asks three questions.

A. Do the same users get a hit?  For every (query arm, lexical scorer) pair: users with a hit
   (exact nDCG@10 > 0) under each, under both, the number of both-hit users expected by chance
   when the two hit sets are independent (n_a * n_b / n) and when they are independent inside
   strata of the number of relevant items of the user (users with many relevant items hit more
   often under any method), the Jaccard index and a hypergeometric p-value (independence only).

B. Are the hit items reachable by words?  An item is REACHABLE for a user when its title shares at
   least one RARE token with the titles of the books of the user's window (the query of the lexical
   scorers). A token is rare when it occurs in at most ``RARE`` (1%) of the catalog titles
   (``--rare`` adds a sensitivity threshold). For every ranking the share of its hit pairs
   (relevant item in the top 10) that are reachable, against two chance levels: the share of the
   user's relevant items that are reachable, and the share of catalog items that are reachable.

C. Semantic residual.  The relevant set of a user is split into reachable items and the RESIDUAL
   items (no shared rare token). Every ranking is scored again, with the same lists, against the
   residual relevant set alone (users without a residual item are left out) and against the
   reachable one. Nothing is re-embedded. The MASKED residual removes each user's reachable items from the catalog before ranking (the
   arms are re-ranked over what is left: conditions ``masked_<arm>``), so residual items no longer
   compete with reachable ones for the top 10. The rare threshold is swept at 0.1%, 1% and 5%.
   Every paired test also reports the minimum detectable effect (``mde80``: alpha 0.05, power 80%,
   from the SD of the paired differences). Paired tests (sign-flip, Holm per domain) compare the
   query arms with each other and with the lexical scorers on the residual set.

Outputs (results/granularity/):
  lexical_overlap_{domain}.csv.gz  per-user TEST metrics under gt = exact / resid / reach (git-ignored)
  lexical_overlap_{domain}.json    parameters, tables A and B, means, provenance, runtime
  stats_lexical_overlap.csv        (--stats) paired tests on the residual relevant set
  summary_lexical_overlap.csv      (--stats) means and hit counts of every ranking under each gt

Usage:
    python -m experiments.granularity.lexical_overlap --domain music
    python -m experiments.granularity.lexical_overlap --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.stats import hypergeom

from acw.corrections import center, zca_power
from acw.evaluation.pipeline import MAX_K
from acw.evaluation.ranking import top_k, top_k_scores_ties
from acw.provenance import provenance
from experiments.granularity import common as C
from experiments.granularity import lexical as L
from experiments.granularity.run_corrections import REGISTRY, score_split

log = logging.getLogger("lexical_overlap")

RARE = 0.01  # a token is rare when it occurs in at most this share of the catalog titles
RARE_SENSITIVITY = (0.001, 0.05)  # the sweep is 0.1%, 1% (main) and 5%
ARMS = ("raw", "center_train_users", "zca_train_users_a1.0")  # query arms of query_norm.py
LEXICAL = ("idf_overlap_title", "bm25_title", L.STRICT_H2)
BASELINES = ("popularity",)
STRATA_EDGES = (2, 3, 5, 10)  # users with 1, 2, 3-4, 5-9 and 10+ relevant items
BATCH = 128  # users per dense reachability matrix
GT_SETS = ("resid", "reach")
# (id, condition, reference) of the tests on the residual relevant set
TESTS = (
    ("zca_vs_raw", "zca_train_users_a1.0", "raw"),
    ("center_vs_raw", "center_train_users", "raw"),
    ("zca_vs_center", "zca_train_users_a1.0", "center_train_users"),
    *((f"zca_vs_{s}", "zca_train_users_a1.0", s) for s in LEXICAL),
    *((f"center_vs_{s}", "center_train_users", s) for s in LEXICAL),
    ("zca_vs_popularity", "zca_train_users_a1.0", "popularity"),
    # masked residual: the reachable items are removed from the catalog before ranking
    ("masked_zca_vs_raw", "masked_zca_train_users_a1.0", "masked_raw"),
    ("masked_center_vs_raw", "masked_center_train_users", "masked_raw"),
    ("masked_zca_vs_center", "masked_zca_train_users_a1.0", "masked_center_train_users"),
)
Z_MDE = 1.959964 + 0.841621  # two-sided alpha 0.05, power 80%


# ─────────────────────────────────────────────────────────────────────────────
# A. Hit overlap
# ─────────────────────────────────────────────────────────────────────────────


def strata_of(n_relevant: np.ndarray) -> np.ndarray:
    """Stratum (0..len(STRATA_EDGES)) of every user from its number of relevant items."""
    return np.digitize(n_relevant, STRATA_EDGES)


def overlap_row(a: np.ndarray, b: np.ndarray, strata: np.ndarray) -> dict:
    """Overlap of two boolean hit vectors over the same users, with the number of both-hit users
    expected by chance when they are independent (overall and inside ``strata``)."""
    n, na, nb, both = len(a), int(a.sum()), int(b.sum()), int((a & b).sum())
    exp_strat = sum(
        a[strata == s].sum() * b[strata == s].sum() / max((strata == s).sum(), 1)
        for s in np.unique(strata)
    )
    return {
        "n": n,
        "n_a": na,
        "n_b": nb,
        "both": both,
        "expected_independent": na * nb / n,
        "expected_stratified": float(exp_strat),
        "jaccard": both / (na + nb - both) if na + nb - both else float("nan"),
        "share_a_in_b": both / na if na else float("nan"),
        "p_hypergeom": float(hypergeom.sf(both - 1, n, na, nb)),
    }


# ─────────────────────────────────────────────────────────────────────────────
# B. Reachability by rare title tokens
# ─────────────────────────────────────────────────────────────────────────────


def rare_items(index: L.LexIndex, frac: float) -> sp.csr_matrix:
    """Binary (tokens x catalog) matrix of the document-token incidences restricted to the rare
    tokens (document frequency <= ``frac`` of the catalog)."""
    n_docs = index.overlap.shape[1]
    df = np.diff(index.overlap.indptr)  # the rows of `overlap` are tokens
    keep = (df <= frac * n_docs).astype(np.float32)
    binary = index.overlap.astype(np.float32)
    binary.data[:] = 1.0
    return (sp.diags(keep) @ binary).tocsr()


def reachable(query_counts: sp.csr_matrix, rare: sp.csr_matrix) -> np.ndarray:
    """Dense boolean (users x catalog): the item shares a rare token with the user's query."""
    q = query_counts.copy().astype(np.float32)
    q.data[:] = 1.0
    return (q @ rare).toarray() > 0


# ─────────────────────────────────────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────────────────────────────────────


def strict_top(pop: C.Population, index: L.LexIndex, queries: list[list[str]]) -> np.ndarray:
    """Top-k of the stricter matcher (IDF overlap, ties by training popularity), as lexical.run."""
    q = index.query(queries)
    tops = []
    for i in range(0, len(queries), L.BATCH):
        sc = index.scores("idf_overlap", q[i : i + L.BATCH])
        sc = sc + L.STRICT_TIE * np.log1p(pop.counts)
        tops.append(top_k_scores_ties(sc, MAX_K, pop.tie_rank))
    return np.vstack(tops)


def anatomy(
    pop: C.Population, users: list[str], tops: dict[str, np.ndarray], rare_by_frac, q_title
) -> tuple[dict, dict]:
    """Per rare threshold: hit-pair shares of every ranking (table B) and the reachable /
    residual relevant sets of every user (ground truths of table C)."""
    rel_idx = [np.array(sorted(pop.item_idx[a] for a in pop.exact_gt[u] if a in pop.item_idx))
               for u in users]  # fmt: skip
    out_b: dict[float, dict] = {}
    sets: dict[float, dict[str, dict[str, set[str]]]] = {}
    for frac, rare in rare_by_frac.items():
        acc = {m: [0, 0, 0] for m in tops}  # hit pairs, reachable hit pairs, reachable top-10 items
        reach_cat, n_rel, n_rel_reach = [], 0, 0
        resid: dict[str, set[str]] = {}
        reach_set: dict[str, set[str]] = {}
        for i in range(0, len(users), BATCH):
            r = reachable(q_title[i : i + BATCH], rare)
            for j in range(r.shape[0]):
                u, rel, row = users[i + j], rel_idx[i + j], r[j]
                flags = row[rel]
                n_rel += len(rel)
                n_rel_reach += int(flags.sum())
                reach_cat.append(row.mean())
                resid[u] = {pop.idx_to_item[k] for k in rel[~flags]}
                reach_set[u] = {pop.idx_to_item[k] for k in rel[flags]}
                relset = set(rel.tolist())
                for m, top in tops.items():
                    t10 = top[i + j, :10]
                    hits = [k for k in t10 if k in relset]
                    acc[m][0] += len(hits)
                    acc[m][1] += int(sum(row[k] for k in hits))
                    acc[m][2] += int(row[t10].sum())
        out_b[frac] = {
            "relevant_items": n_rel,
            "relevant_reachable_share": n_rel_reach / n_rel,
            "catalog_reachable_share": float(np.mean(reach_cat)),
            "rankings": {
                m: {
                    "hit_pairs": a[0],
                    "hit_pairs_reachable": a[1],
                    "hit_reachable_share": a[1] / a[0] if a[0] else float("nan"),
                    "top10_reachable_share": a[2] / (10 * len(users)),
                }
                for m, a in acc.items()
            },
            "users_with_residual": sum(1 for s in resid.values() if s),
            "users_with_reachable": sum(1 for s in reach_set.values() if s),
        }
        sets[frac] = {"resid": resid, "reach": reach_set}
    return out_b, sets


def masked_tops(
    queries: dict[str, np.ndarray], items: np.ndarray, q_title, rare, tie_rank: np.ndarray
) -> dict[str, np.ndarray]:
    """Top-k of every query arm over the catalog WITHOUT the user's reachable items (those sharing
    a rare title token with the user's books): the residual items no longer compete with them for
    the top 10, so the crowd-out of the unmasked residual score is gone."""
    out: dict[str, list[np.ndarray]] = {a: [] for a in queries}
    n = next(iter(queries.values())).shape[0]
    for i in range(0, n, BATCH):
        r = reachable(q_title[i : i + BATCH], rare)
        for arm, q in queries.items():
            sc = q[i : i + BATCH] @ items.T
            sc[r] = -np.inf
            out[arm].append(top_k_scores_ties(sc, MAX_K, tie_rank))
    return {f"masked_{a}": np.vstack(v) for a, v in out.items()}


def mde80(diff: np.ndarray) -> float:
    """Minimum detectable mean difference (alpha 0.05 two-sided, power 80%) of a paired test
    from the standard deviation of the per-user differences."""
    d = np.asarray(diff, float)
    return float(Z_MDE * d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else float("nan")


def run(domain: str) -> None:
    t0 = time.time()
    src = REGISTRY[(domain, "gte", "titlecat")]
    pop = C.build_population(domain, src)
    emb = C.load_embeddings(src, pop, sources=("train_users",), shrinkage=False)
    mom = emb.moments["train_users"]
    users = pop.eval_users["test"]
    q = emb.eval_prof["test"]
    arm_q = {"raw": q, "center_train_users": center(q, mom.mean),
             "zca_train_users_a1.0": zca_power(q, *mom, 1.0)}  # fmt: skip
    tops = {arm: top_k(x, emb.items, MAX_K) for arm, x in arm_q.items()}
    indexes = L.build_indexes(pop)
    titles = L.book_titles(L.BOOKS_PATH, L.h2_window_books(pop))
    queries = L.user_queries(pop, users, titles)
    tops["idf_overlap_title"] = L.lexical_top(
        indexes["title"], "idf_overlap", queries, pop.tie_rank
    )
    tops["bm25_title"] = L.lexical_top(indexes["title"], "bm25", queries, pop.tie_rank)
    strict_q = L.user_queries(pop, users, titles, L.strict_tokens)
    tops[L.STRICT_H2] = strict_top(pop, indexes["strict"], strict_q)
    pop_list = [pop.item_idx[a] for a in pop.constants["exact"]["popularity"][:MAX_K]]
    tops["popularity"] = np.tile(np.array(pop_list), (len(users), 1))
    log.info(f"{domain}: ranked {list(tops)} for {len(users)} test users")

    # Exact-level scores; the arms and the baselines must reproduce run_corrections
    ranked = {k: v for k, v in tops.items() if k != "popularity"}
    exact = C.score(pop, "test", ranked)
    check = C.check_against_grid(
        exact,
        f"{domain}_gte_titlecat",
        "test",
        {"raw": "raw", "zca_train_users_a1.0": "zca_profile"},
    )

    # A. hit overlap
    wide = exact[exact["gt"] == "exact"].pivot(index="user_id", columns="condition",
                                               values="nDCG@10").loc[users]  # fmt: skip
    hit = (wide > 0).to_dict("series")
    strata = strata_of(np.array([len(pop.exact_gt[u]) for u in users]))
    table_a = {
        arm: {lex: overlap_row(hit[arm].to_numpy(), hit[lex].to_numpy(), strata) for lex in LEXICAL}
        for arm in ARMS
    }

    # B. reachability of the hit pairs and C. the residual relevant sets
    fracs = (RARE, *RARE_SENSITIVITY)
    index = indexes["title"]
    rare_by_frac = {f: rare_items(index, f) for f in fracs}
    q_title = index.query(queries)
    table_b, sets = anatomy(pop, users, tops, rare_by_frac, q_title)
    constants = {**pop.constants, **{g: pop.constants["exact"] for g in GT_SETS}}
    frames = [exact]
    for frac in fracs:
        extra = sets[frac]
        masked = masked_tops(arm_q, emb.items, q_title, rare_by_frac[frac], pop.tie_rank)
        part, *_ = score_split("test", users, {**ranked, **masked}, constants, {}, None,
                               pop.exact_gt, pop.cfg, pop.idx_to_item, True,
                               extra_gt=extra)  # fmt: skip
        part = pd.concat(part[1:], ignore_index=True)  # drop the exact frame (already scored)
        part["gt"] = part["gt"] + ("" if frac == RARE else f"@{frac:g}")
        frames.append(part)
    per_user = pd.concat(frames, ignore_index=True)
    summary = {
        "domain": domain,
        "encoder": "gte",
        "text": "titlecat",
        "rare_fraction": RARE,
        "rare_sensitivity": list(RARE_SENSITIVITY),
        "n_catalog": len(pop.idx_to_item),
        "n_test_users": len(users),
        "arms": list(ARMS),
        "lexical": list(LEXICAL),
        "strata_edges": list(STRATA_EDGES),
        "n_users_per_stratum": np.bincount(strata).tolist(),
        "A_hit_overlap": table_a,
        "B_reachability": {f"{f:g}": v for f, v in table_b.items()},
        "test": C.block_means(per_user),
        "grid_check_max_abs_diff": check,
        "provenance": C.provenance_of(pop, *emb.files, L.BOOKS_PATH),
        **C.run_info(t0),
    }
    C.write_run(f"lexical_overlap_{domain}", per_user, summary)


# ─────────────────────────────────────────────────────────────────────────────
# Paired statistics
# ─────────────────────────────────────────────────────────────────────────────


def stats() -> None:
    rows, summ_rows, files = [], [], []
    for domain in C.DOMAINS:
        stem = f"lexical_overlap_{domain}"
        path = C.OUT_DIR / f"{stem}.json"
        if not path.exists():
            log.warning(f"{stem}: no result, skipped")
            continue
        files.append(C.OUT_DIR / f"{stem}.csv.gz")
        summ = json.loads(path.read_text())
        df = pd.read_csv(C.OUT_DIR / f"{stem}.csv.gz")
        for gt, block in summ["test"].items():
            for cond, m in block.items():
                summ_rows.append({"domain": domain, "gt": gt, "condition": cond, **m})
        for gt in sorted(set(df["gt"]) - {"exact"}):
            sub = df[df["gt"] == gt]
            for metric, short in C.STAT_METRICS.items():
                wide = sub.pivot(index="user_id", columns="condition", values=metric).sort_index()
                wide = wide.dropna(axis=0)
                meta = {"domain": domain, "encoder": "gte", "text": "titlecat", "gt": gt,
                        "metric": short}  # fmt: skip
                pairs = [
                    t
                    for t in TESTS
                    if t[1] in wide
                    and t[2] in wide
                    and (gt.startswith("resid") or not t[0].startswith("masked"))
                ]
                new = C.paired_rows(wide, pairs, meta, f"residual:{gt}")
                for r, (_, c, ref) in zip(new, pairs):
                    r["mde80"] = mde80(wide[c].to_numpy() - wide[ref].to_numpy())
                rows += new
        log.info(f"{stem}: tested")
    pd.DataFrame(rows).to_csv(C.OUT_DIR / "stats_lexical_overlap.csv", index=False,
                              float_format="%.6g")  # fmt: skip
    pd.DataFrame(summ_rows).to_csv(C.OUT_DIR / "summary_lexical_overlap.csv", index=False,
                                   float_format="%.6g")  # fmt: skip
    prov = {"provenance": provenance(files)}
    (C.OUT_DIR / "stats_lexical_overlap_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info(f"saved stats_lexical_overlap.csv ({len(rows)} rows) and summary_lexical_overlap.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=C.DOMAINS)
    ap.add_argument("--stats", action="store_true", help="paired tests over the finished runs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.stats:
        stats()
    elif args.domain:
        run(args.domain)
    else:
        ap.error("give --domain or --stats")


if __name__ == "__main__":
    main()
