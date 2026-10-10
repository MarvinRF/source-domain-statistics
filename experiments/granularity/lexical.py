"""
Embedding-free lexical baselines for item-level cross-domain recommendation.

For a user the query is the concatenation of the titles of the books that enter
the H2 profile (the last MAX_BOOKS_PROFILE embedded books of the history); each
target item is a document made of

  title      the item title only
  titlecat   the item title followed by its category path (the text that the
             title+categories encoders embed)

Documents and queries are lower-cased and split into runs of letters and digits
(``[^\\W_]+``); there is no stop list and no stemming. Scorers over the whole
target catalog:

  bm25          Okapi BM25, k1 = 1.2, b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5)),
                the query term frequency counts (a title token repeated across the
                books weighs more)
  idf_overlap   sum of idf over the DISTINCT tokens shared by the query and the document

N and df come from the target catalog documents. Ties (most items score zero) are
broken by ascending item id, so the lists use no interaction log. The hybrid adds
``lambda`` times the train popularity z-score (as experiments.granularity.hybrid)
to the score divided, per user, by its maximum over the catalog (0 for a user
without any matching token); lambda comes from the grid of hybrid.py and is chosen
per scorer and text on the VAL users by mean exact nDCG@10 (ties: the smaller
lambda), and lambda = inf ranks by popularity with ties broken by the lexical score.

The lexical scores do not depend on the embedding text: ``--text`` only names the run (suffix
``_<text>``) and pairs it, in ``--stats``, with the embedding run of the same text; the results
of title / titlecat are identical. ``--text desc`` (music only) instead scores the long text
(title + categories + description, as the ``desc`` encoder run embeds it): bases ``bm25_desc`` and
``idf_overlap_desc``, paired with query_norm_music_gte_desc.

Outputs (results/granularity/):
  lexical_{domain}.csv.gz   per-user TEST metrics of the baselines, the four lexical
                            scorers and their selected hybrids (git-ignored)
  lexical_{domain}.json     lambda grid, VAL curve (selected on it), TEST curve
                            (diagnostic only), selected lambda, VAL and TEST means,
                            provenance, runtime and peak RSS
  stats_lexical.csv  (--stats)  paired tests on TEST against the gte conditions of
                            query_norm and hybrid; Holm per (domain, metric, family)
  summary_lexical.csv (--stats) TEST means of every condition and domain

Usage:
    python -m experiments.granularity.lexical --domain music
    python -m experiments.granularity.lexical --domain music --text title
    python -m experiments.granularity.lexical --stats
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from acw.evaluation.pipeline import MAX_K
from acw.evaluation.ranking import top_k_scores_ties
from acw.provenance import provenance
from acw.settings import MAX_BOOKS_PROFILE, PROCESSED_DIR, RAW_DIR
from experiments.granularity import common as C
from experiments.granularity.hybrid import LAMBDAS, lam_label
from experiments.granularity.run_corrections import REGISTRY, window_books

log = logging.getLogger("lexical")

K1, B = 1.2, 0.75
BATCH = 256  # users per dense score matrix
TEXTS = ("title", "titlecat")
SCORERS = ("bm25", "idf_overlap")
BASE_SPECS = [(f"{s}_{t}", s, t) for s in SCORERS for t in TEXTS]  # (name, scorer, text)
BASES = tuple(b for b, _, _ in BASE_SPECS)
# long text (music only, the only domain with descriptions): the title + categories + description
# string that the `desc` encoder run embeds (data/processed/desc/albums_texts.json); a `--text desc`
# run scores only these two bases, the title / titlecat ones stay in the other runs
DESC_SPECS = [(f"{s}_desc", s, "desc") for s in SCORERS]
DESC_TEXTS_PATH = PROCESSED_DIR / "desc" / "albums_texts.json"
TOKEN = re.compile(r"[^\W_]+")
BOOKS_PATH = RAW_DIR / "books_filtered.jsonl"


# Tokenisation of the stricter embedding-free matcher (a second, independent implementation that
# checks the pipeline and breaks ties by training popularity): lower-case letters only, at least
# 4 of them, a short stop list, idf = ln(N / (1 + df))
STRICT_TOKEN = re.compile(r"[a-z]{4,}")
STRICT_STOP = frozenset(
    "with from that this your the and for new edition volume vol book books box set best "
    "collection original complete live greatest hits music series season story guide life "
    "world time first".split()
)
STRICT_FULL = "strict_idf_full_window"  # query built from ALL the last history entries
STRICT_H2 = "strict_idf_h2_window"  # the same scoring on the H2 window of this module
STRICT_TIE = 1e-6  # weight of log(1 + n_train) that breaks the ties of the strict score


def tokens(text: str) -> list[str]:
    return TOKEN.findall(text.lower())


def strict_tokens(text: str) -> list[str]:
    return [w for w in STRICT_TOKEN.findall(text.lower()) if w not in STRICT_STOP]


class LexIndex:
    """Sparse BM25 and IDF-overlap weights of the documents of one text variant."""

    def __init__(self, docs: list[list[str]], strict_idf: bool = False):
        self.vocab: dict[str, int] = {}
        rows, cols = [], []
        for r, doc in enumerate(docs):
            for tok, c in Counter(doc).items():
                rows.append((r, c))
                cols.append(self.vocab.setdefault(tok, len(self.vocab)))
        r = np.array([x[0] for x in rows])
        tf = np.array([x[1] for x in rows], dtype=np.float64)
        n, v = len(docs), len(self.vocab)
        c = np.array(cols)
        df = np.bincount(c, minlength=v)
        idf = np.log(n / (1 + df)) if strict_idf else np.log1p((n - df + 0.5) / (df + 0.5))
        dl = np.array([len(d) for d in docs], dtype=np.float64)
        denom = tf + K1 * (1 - B + B * dl[r] / dl.mean())
        # transposed (V x N) so that  query counts (b x V) @ weights  scores the whole catalog
        self.bm25 = sp.csr_matrix((idf[c] * tf * (K1 + 1) / denom, (c, r)), shape=(v, n))
        self.overlap = sp.csr_matrix((idf[c], (c, r)), shape=(v, n))

    def query(self, queries: list[list[str]]) -> sp.csr_matrix:
        """Query term counts (n_queries x V); tokens outside the vocabulary are dropped."""
        indptr, indices, data = [0], [], []
        for q in queries:
            counts = Counter(t for t in q if t in self.vocab)
            indices += [self.vocab[t] for t in counts]
            data += list(counts.values())
            indptr.append(len(indices))
        return sp.csr_matrix(
            (np.array(data, dtype=np.float64), indices, indptr),
            shape=(len(queries), len(self.vocab)),
        )

    def scores(self, scorer: str, q: sp.csr_matrix) -> np.ndarray:
        """Dense (n_queries x N) scores of the catalog in catalog row order."""
        if scorer == "bm25":
            return (q @ self.bm25).toarray()
        binary = q.copy()
        binary.data[:] = 1.0
        return (binary @ self.overlap).toarray()


def build_indexes(pop: C.Population, text: str = "titlecat") -> dict[str, LexIndex]:
    """Title-only and title+categories indexes over the target catalog, rows in catalog order."""
    n = len(pop.idx_to_item)
    title, titlecat, strict = [None] * n, [None] * n, [None] * n
    long = None
    if text == "desc":
        long = json.loads(DESC_TEXTS_PATH.read_text())
    with open(pop.cfg.items_meta_path) as f:
        for line in f:
            item = json.loads(line)
            i = pop.item_idx.get(item.get("parent_asin"))
            if i is None:
                continue
            t = (item.get("title") or "").strip()
            cats = item.get("categories") or []
            cats = " ".join(str(c) for c in cats if c) if isinstance(cats, list) else str(cats)
            title[i], titlecat[i] = tokens(t), tokens(f"{t} {cats}".strip())
            strict[i] = strict_tokens(t)
    assert all(d is not None for d in title), "catalog items without metadata"
    extra = {}
    if long is not None:
        docs = [tokens(long[pop.idx_to_item[i]]) for i in range(n)]
        extra["desc"] = LexIndex(docs)
    return {
        **extra,
        "title": LexIndex(title),
        "titlecat": LexIndex(titlecat),
        "strict": LexIndex(strict, strict_idf=True),
    }


def user_queries(
    pop: C.Population, users: list[str], titles: dict[str, str], tokenize=tokens
) -> list[list[str]]:
    """Tokens of the concatenated titles of the books of the H2 window of every user."""
    return [
        [t for a in window_books(pop.history.get(u, []), pop.book_idx) for t in tokenize(titles[a])]
        for u in users
    ]


def book_titles(path: Path, need: set[str]) -> dict[str, str]:
    """Titles (``""`` when missing from the file) of the books ``need`` in a books jsonl."""
    out = dict.fromkeys(need, "")
    with open(path) as f:
        for line in f:
            item = json.loads(line)
            if item.get("parent_asin") in need:
                out[item["parent_asin"]] = item.get("title") or ""
    return out


def h2_window_books(pop: C.Population) -> set[str]:
    return {
        a
        for users in pop.eval_users.values()
        for u in users
        for a in window_books(pop.history.get(u, []), pop.book_idx)
    }


def lexical_top(
    index: LexIndex, scorer: str, queries: list[list[str]], tie_rank: np.ndarray, k: int = MAX_K
) -> np.ndarray:
    """Pure lexical ranking (ties by ``tie_rank``) of every query."""
    q = index.query(queries)
    tops = [
        top_k_scores_ties(index.scores(scorer, q[i : i + BATCH]), k, tie_rank)
        for i in range(0, len(queries), BATCH)
    ]
    return np.vstack(tops)


def lexical_tops(
    index: LexIndex, scorer: str, queries: list[list[str]], pop: C.Population
) -> dict[float, np.ndarray]:
    """Top-k of ``lexical/max + lambda * pz`` for every lambda of the grid."""
    q = index.query(queries)
    out: dict[float, list[np.ndarray]] = {lam: [] for lam in LAMBDAS}
    for i in range(0, len(queries), BATCH):
        s = index.scores(scorer, q[i : i + BATCH])
        peak = s.max(axis=1, keepdims=True)
        s = np.divide(s, peak, out=np.zeros_like(s), where=peak > 0)
        for lam, top in C.lambda_tops(s, pop.pz, pop.tie_rank, MAX_K).items():
            out[lam].append(top)
    return {lam: np.vstack(v) for lam, v in out.items()}


def run(domain: str, text: str = "titlecat") -> None:
    t0 = time.time()
    src = REGISTRY[(domain, "gte", text)]  # only its id maps are read: no embedding is loaded
    pop = C.build_population(domain, src)
    indexes = build_indexes(pop, text)
    base_specs = DESC_SPECS if text == "desc" else BASE_SPECS
    bases = tuple(b for b, _, _ in base_specs)
    titles = book_titles(BOOKS_PATH, h2_window_books(pop))
    queries = {s: user_queries(pop, u, titles) for s, u in pop.eval_users.items()}
    n_empty = {s: sum(1 for q in qs if not q) for s, qs in queries.items()}
    log.info(
        f"indexes: { {t: len(i.vocab) for t, i in indexes.items()} } tokens; empty queries {n_empty}"
    )

    rankings: dict[str, dict[str, np.ndarray]] = {s: {} for s in pop.eval_users}
    for base, scorer, doc in base_specs:
        for s in pop.eval_users:
            for lam, top in lexical_tops(indexes[doc], scorer, queries[s], pop).items():
                rankings[s][f"{base}|{lam_label(lam)}"] = top
        log.info(f"ranked {base}")

    # Pipeline check: a stricter IDF-overlap matcher (letters-only tokens of at least
    # four characters, short stop list, idf = ln(N / (1 + df)), ties broken by train popularity).
    # STRICT_FULL queries with ALL of the last MAX_BOOKS_PROFILE history entries, titles from the
    # domain's books file (an unknown book has no title); STRICT_H2 uses the H2 window instead.
    dom_books = pop.cfg.raw_dir / "books_filtered.jsonl"
    last = {
        s: [pop.history.get(u, [])[-MAX_BOOKS_PROFILE:] for u in users]
        for s, users in pop.eval_users.items()
    }
    dom_titles = book_titles(dom_books, {a for ls in last.values() for books in ls for a in books})
    rev_q = {
        STRICT_FULL: {
            s: [[t for a in books for t in strict_tokens(dom_titles[a])] for books in ls]
            for s, ls in last.items()
        },
        STRICT_H2: {
            s: user_queries(pop, u, titles, strict_tokens) for s, u in pop.eval_users.items()
        },
    }
    check: dict[str, dict[str, np.ndarray]] = {}
    for name, per_split in rev_q.items():
        check[name] = {}
        for s, qs in per_split.items():
            q = indexes["strict"].query(qs)
            tops = []
            for i in range(0, len(qs), BATCH):
                sc = indexes["strict"].scores("idf_overlap", q[i : i + BATCH])
                sc = sc + STRICT_TIE * np.log1p(pop.counts)
                tops.append(top_k_scores_ties(sc, MAX_K, pop.tie_rank))
            check[name][s] = np.vstack(tops)
    val = C.score(pop, "val", {**rankings["val"], **{n: c["val"] for n, c in check.items()}})
    test_curve = C.score(pop, "test", rankings["test"])
    chosen = {b: C.select_lambda(val, b) for b in bases}
    log.info(f"selected lambda: { {b: lam_label(v) for b, v in chosen.items()} }")
    keep: dict[str, np.ndarray] = {}
    rename: dict[str, str] = {}
    for b in bases:
        keep[b] = rankings["test"][f"{b}|0"]
        keep[f"hybrid_{b}"] = rankings["test"][f"{b}|{lam_label(chosen[b])}"]
        rename[f"{b}|0"], rename[f"{b}|{lam_label(chosen[b])}"] = b, f"hybrid_{b}"
    keep.update({n: c["test"] for n, c in check.items()})
    test = C.score(pop, "test", keep, d2=True, strata=True)
    val_sel = val[~val["condition"].str.contains(r"\|") | val["condition"].isin(rename)].copy()
    val_sel["condition"] = val_sel["condition"].replace(rename)

    def curve(df: pd.DataFrame) -> dict:
        m = C.block_means(df)["exact"]
        return {b: {lam_label(lam): m[f"{b}|{lam_label(lam)}"] for lam in LAMBDAS} for b in bases}

    summary = {
        "domain": domain,
        "text": text,
        "tokenisation": "lower case, runs of letters and digits, no stop list, no stemming",
        "bm25": {"k1": K1, "b": B, "idf": "ln(1 + (N - df + 0.5) / (df + 0.5))"},
        "bases": list(bases),
        "score": "lexical score / max over the catalog + lambda * z(log(1 + n_train_users(item)))",
        "lambdas": [lam_label(lam) for lam in LAMBDAS],
        "selection": "VAL users, mean exact nDCG@10, ties to the smaller lambda",
        "selected_lambda": {b: lam_label(v) for b, v in chosen.items()},
        "n_eval_users": {s: len(u) for s, u in pop.eval_users.items()},
        "n_empty_queries": n_empty,
        "vocabulary": {t: len(i.vocab) for t, i in indexes.items()},
        "val_curve": curve(val),
        "test_curve_diagnostic": curve(test_curve),
        "val": C.block_means(val_sel),
        "test": C.block_means(test),
        "provenance": C.provenance_of(pop, BOOKS_PATH, dom_books),
        **C.run_info(t0),
    }
    C.write_run(f"lexical_{domain}{C.text_tag(text)}", test, summary)


# ─────────────────────────────────────────────────────────────────────────────
# Paired statistics
# ─────────────────────────────────────────────────────────────────────────────

QN_REFS = ("raw", "center_train_users", "zca_train_users_a1.0")  # conditions of query_norm (gte)
HY_REFS = (
    "hybrid_raw",
    "hybrid_zca_profile",
    "hybrid_zca_catalog_a1.0",
)  # conditions of hybrid (gte)


def stats() -> None:
    rows, summ_rows, files = [], [], []
    for domain, text in C.gte_runs():
        tag = C.text_tag(text)
        stem = f"lexical_{domain}{tag}"
        if not (C.OUT_DIR / f"{stem}.json").exists():
            log.warning(f"{stem}: no result, skipped")
            continue
        summ = json.loads((C.OUT_DIR / f"{stem}.json").read_text())
        summ_rows += C.flat_rows(summ)
        # the popularity hybrids of the embeddings exist for the title+categories text only
        hy_refs = HY_REFS if text == "titlecat" else ()
        qn_stem, hy_stem = f"query_norm_{domain}_gte{tag}", f"hybrid_{domain}_gte"
        used = [stem, qn_stem, *([hy_stem] if hy_refs else [])]
        files += [C.OUT_DIR / f"{s}.csv.gz" for s in used]
        wides = {s: C.wide_test(s) for s in used}
        for metric in wides[stem]:
            parts = [wides[stem][metric], wides[qn_stem][metric][list(QN_REFS)]]
            if hy_refs:
                parts.append(wides[hy_stem][metric][list(hy_refs)])
            assert all(p.index.equals(parts[0].index) for p in parts)
            wide = pd.concat(parts, axis=1)
            meta = {"domain": domain, "encoder": "gte", "text": text, "metric": metric}
            bases = [b for b, _, _ in (DESC_SPECS if text == "desc" else BASE_SPECS)]
            pure = [(f"{b}_vs_{r}", b, r) for b in bases for r in (*QN_REFS, "popularity")]
            hybrid = [
                (f"hybrid_{b}_vs_{r}", f"hybrid_{b}", r)
                for b in bases
                for r in ("popularity", b, *hy_refs)
            ]
            rows += C.paired_rows(wide, pure, meta, "lexical_vs_content")
            rows += C.paired_rows(wide, hybrid, meta, "lexical_hybrid")
        log.info(f"{stem}: tested")
    df = pd.DataFrame(rows)
    df.to_csv(C.OUT_DIR / "stats_lexical.csv", index=False, float_format="%.6g")
    pd.DataFrame(summ_rows).to_csv(
        C.OUT_DIR / "summary_lexical.csv", index=False, float_format="%.6g"
    )
    prov = {"provenance": provenance(files)}
    (C.OUT_DIR / "stats_lexical_provenance.json").write_text(json.dumps(prov, indent=2))
    log.info(f"saved stats_lexical.csv ({len(df)} rows) and summary_lexical.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=C.DOMAINS)
    ap.add_argument("--text", choices=C.ITEM_TEXTS, default="titlecat")
    ap.add_argument("--stats", action="store_true", help="paired tests over the finished runs")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")
    if args.stats:
        stats()
    elif args.domain:
        run(args.domain, args.text)
    else:
        ap.error("give --domain or --stats")


if __name__ == "__main__":
    main()
