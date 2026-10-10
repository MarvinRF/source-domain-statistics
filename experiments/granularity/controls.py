"""
Shuffled-profile control: is the item-level signal personal?

The H2 profiles (and, for the lexical baseline, the queries) of the TEST users are
permuted among the TEST users with a seeded derangement (``acw.settings.SEED``: no
user keeps its own profile) while every user keeps their own relevant items. The
fitted statistics (train-user moments, catalog moments) are unchanged, so a shuffled
condition is the real one evaluated on someone else's profile. Conditions: ``raw``,
``center_train_users``, ``zca_train_users_a1.0``, ``zca_catalog_a1.0`` and the BM25
matcher over the title and over the title + categories.

Per-user TEST metrics of the real and the shuffled (``shuffled_<condition>``)
conditions are scored under the exact ground truth, depth-2 categories and the cold /
warm subsets, together with the baselines.

Outputs (results/granularity/; a text other than titlecat adds the suffix _<text>):
  controls_{domain}_{encoder}.csv.gz  per-user TEST metrics (git-ignored)
  controls_{domain}_{encoder}.json    seed, TEST means, grid check, provenance
  stats_controls.csv   (--stats)  shuffled vs real, per condition; Holm per
                                  (domain, metric)
  summary_controls.csv (--stats)  TEST means of every condition

Usage:
    python -m experiments.granularity.controls --domain music --encoder gte
    python -m experiments.granularity.controls --domain music --encoder gte --text title
    python -m experiments.granularity.controls --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import time

import numpy as np
import pandas as pd

from acw.corrections import center, zca_power
from acw.evaluation.pipeline import MAX_K
from acw.evaluation.ranking import top_k
from acw.provenance import provenance
from acw.settings import SEED
from experiments.granularity import common as C
from experiments.granularity.lexical import (
    BOOKS_PATH,
    book_titles,
    build_indexes,
    h2_window_books,
    lexical_top,
    user_queries,
)
from experiments.granularity.query_norm import GRID_EQUIV
from experiments.granularity.run_corrections import REGISTRY

log = logging.getLogger("controls")

CONDITIONS = (
    "raw",
    "center_train_users",
    "zca_train_users_a1.0",
    "zca_catalog_a1.0",
    "bm25_title",
    "bm25_titlecat",
)
ENCODER_SOURCES = ("train_users", "target_catalog")


def derangement(n: int, seed: int = SEED) -> np.ndarray:
    """Seeded permutation of ``range(n)`` without fixed points (n >= 2)."""
    perm = np.random.default_rng(seed).permutation(n)
    while (fixed := np.flatnonzero(perm == np.arange(n))).size:
        i = fixed[0]
        j = (i + 1) % n  # swapping a fixed point with its neighbour fixes neither of them
        perm[i], perm[j] = perm[j], perm[i]
    return perm


def run(domain: str, encoder: str, text: str = "titlecat") -> None:
    t0 = time.time()
    src = REGISTRY[(domain, encoder, text)]
    pop = C.build_population(domain, src)
    emb = C.load_embeddings(src, pop, sources=ENCODER_SOURCES, shrinkage=False)
    mom_u, mom_c = emb.moments["train_users"], emb.moments["target_catalog"]
    q = emb.eval_prof["test"]
    zca_items = zca_power(emb.items, *mom_c, 1.0)
    real = {
        "raw": top_k(q, emb.items, MAX_K),
        "center_train_users": top_k(center(q, mom_u.mean), emb.items, MAX_K),
        "zca_train_users_a1.0": top_k(zca_power(q, *mom_u, 1.0), emb.items, MAX_K),
        "zca_catalog_a1.0": top_k(q, zca_items, MAX_K),
    }
    del zca_items
    indexes = build_indexes(pop)
    titles = book_titles(BOOKS_PATH, h2_window_books(pop))
    queries = user_queries(pop, pop.eval_users["test"], titles)
    for doc in ("title", "titlecat"):  # item text of the lexical documents (not the embedded text)
        real[f"bm25_{doc}"] = lexical_top(indexes[doc], "bm25", queries, pop.tie_rank)
    assert tuple(real) == CONDITIONS
    perm = derangement(len(q))
    rankings = {c: top for c, top in real.items()}
    rankings.update({f"shuffled_{c}": top[perm] for c, top in real.items()})
    test = C.score(pop, "test", rankings, d2=True, strata=True)
    grid_check = C.check_against_grid(
        test,
        f"{domain}_{encoder}_{text}",
        "test",
        {k: v for k, v in GRID_EQUIV.items() if k in real},
    )
    summary = {
        "domain": domain,
        "encoder": encoder,
        "model": src.model,
        "text": text,
        "seed": SEED,
        "shuffle": "seeded derangement of the TEST users' profiles / queries",
        "conditions": list(CONDITIONS),
        "n_eval_users": {s: len(u) for s, u in pop.eval_users.items()},
        "test": C.block_means(test),
        "val": {},
        "grid_check_max_abs_diff": grid_check,
        "provenance": C.provenance_of(pop, *emb.files),
        **C.run_info(t0),
    }
    C.write_run(f"controls_{domain}_{encoder}{C.text_tag(text)}", test, summary)


def stats(encoder: str = "gte") -> None:
    rows, summ_rows, files = [], [], []
    for domain, text in C.gte_runs():
        stem = f"controls_{domain}_{encoder}{C.text_tag(text)}"
        path = C.OUT_DIR / f"{stem}.json"
        if not path.exists():
            log.warning(f"{stem}: no result, skipped")
            continue
        summ_rows += C.flat_rows(json.loads(path.read_text()))
        files.append(C.OUT_DIR / f"{stem}.csv.gz")
        for metric, wide in C.wide_test(stem).items():
            meta = {"domain": domain, "encoder": encoder, "text": text, "metric": metric}
            pairs = [(f"shuffled_{c}_vs_{c}", f"shuffled_{c}", c) for c in CONDITIONS]
            rows += C.paired_rows(wide, pairs, meta, "shuffled_vs_real")
        log.info(f"{stem}: tested")
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / "stats_controls.csv", index=False, float_format="%.6g")
    pd.DataFrame(summ_rows).to_csv(
        C.OUT_DIR / "summary_controls.csv", index=False, float_format="%.6g"
    )
    prov = {"provenance": provenance(files)}
    (C.OUT_DIR / "stats_controls_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info(f"saved stats_controls.csv ({len(df)} rows) and summary_controls.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=C.DOMAINS)
    ap.add_argument("--encoder", default="gte")
    ap.add_argument("--text", choices=C.ITEM_TEXTS, default="titlecat")
    ap.add_argument("--stats", action="store_true", help="paired tests over the finished runs")
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
