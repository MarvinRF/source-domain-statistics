"""
LaTeX tables of the paper, generated from the result files only.

Reads ``results/granularity/`` (summary_test.csv, stats_*.csv, one JSON per run, the
``hybrid_*_gte.json`` and ``stats_hybrid.csv`` of experiments.granularity.hybrid, the
``summary_`` / ``stats_`` / ``tally_`` files of the query-normalisation (``query_norm``),
``lexical``, ``knn`` and ``controls`` experiments, ``taxonomy.json``, and, for the tables of
users with a hit, the per-user ``*_gte_titlecat.csv.gz`` of the three target domains) and
``results/mechanism/``. Table 1 also reads the user counts of ``data/splits/{domain}_stats.json``
and the interaction means of ``data/raw/**/preprocessing_stats.json``. Nothing is hard-coded
except labels and the embedding provenance of the music encoders (transcribed from
``docs/EMBEDDINGS.md``, see ``ENCODERS``).

Each file written to ``--out`` holds one ``tabular`` (no caption, no label); the
paper wraps it with ``\\input``. Markers: an up (down) arrow means the
correction is better (worse) than ``raw`` with Holm-adjusted p < 0.05 of the PRIMARY test
(the paired sign-flip test on the mean, ``p_holm_signflip``) inside the family of corrections
of its (domain, encoder, text, granularity, metric), PCA whitening excluded; a
star means the same for a signed relative gain; a dagger means that the 95%
bootstrap interval of a granularity gap excludes zero. Baselines (random,
popularity, top-category) are outside the Holm family and carry no marker. Every table of
paired tests reports the number of users whose two values differ (``n_nonzero_diff``).

Usage (repository root):
    python -m experiments.figures.paper_tables --out "<paper dir>/tables"
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from acw.settings import RESULTS_DIR, ROOT

GRAN = RESULTS_DIR / "granularity"
MECH = RESULTS_DIR / "mechanism"
DOMAINS = ("music", "movies", "games")
DNAME = {"music": "Music", "movies": "Movies", "games": "Games"}
DOMAINS2 = ("movies2music", "movies2games")  # movies as SOURCE (source2 files)
DNAME2 = {"movies2music": r"Movies$\to$Music", "movies2games": r"Movies$\to$Games"}
ALPHA = 0.05
UP, DOWN, STAR, DAGGER = r"$^{\uparrow}$", r"$^{\downarrow}$", r"$^{*}$", r"$^{\dagger}$"

# Item-text provenance of the music encoders, transcribed from docs/EMBEDDINGS.md:
# (label, verified, text embedded). TC = title + categories; LT = long text field of
# the filtered catalog (title, description, categories, features; at most 1000
# characters); "?" = the text cannot be recovered.
ENCODERS = {
    "gte": ("GTE-large-v1.5", True, "TC"),
    "bge-large": ("BGE-large-v1.5", True, "LT"),
    "bge-m3": ("BGE-M3", True, "LT"),
    "e5-large-v2": ("E5-large-v2", True, "LT"),
    "minilm": ("MiniLM-L6-v2", True, "LT"),
    "minilm-l12": ("MiniLM-L12-v2", False, "?"),
    "nomic": ("Nomic-embed-v1.5", True, "LT"),
    "qwen3": ("Qwen3-Embedding-8B", False, "TC$^{a}$"),
    "snowflake-arctic-embed-l": ("Arctic-embed-L", True, "LT"),
    "uae-large-v1": ("UAE-Large-v1", True, "LT"),
}
ENC_LABEL = {k: v[0] for k, v in ENCODERS.items()}

COND = {
    "random": "Random",
    "popularity": "Popularity",
    "top_category": "Top-category list",
    "raw": "Raw",
    "center_catalog": "Centering",
    "abtt_catalog_D1": r"ABTT ($D{=}1$)",
    "abtt_catalog_D3": r"ABTT ($D{=}3$)",
    "abtt_catalog_D10": r"ABTT ($D{=}10$)",
    "pca_catalog": "PCA whitening",
    "zca_catalog_a0.25": r"ZCA ($\alpha{=}0.25$)",
    "zca_catalog_a0.5": r"ZCA ($\alpha{=}0.5$)",
    "zca_catalog_a0.75": r"ZCA ($\alpha{=}0.75$)",
    "zca_catalog_a1.0": r"ZCA ($\alpha{=}1$)",
    "zca_catalog_books": r"ZCA, $\Sigma$ of books",
    "zca_catalog_grocery": r"ZCA, $\Sigma$ of grocery",
    "zca_profile": "ZCA (users)",
    "center_profile": "Centring (users)",
    "zca_both_separate": r"ZCA, separate $\Sigma$",
    "zca_symmetric": "ZCA, symmetric",
}
CATALOG_ALL = [
    "center_catalog",
    "abtt_catalog_D1",
    "abtt_catalog_D3",
    "abtt_catalog_D10",
    "pca_catalog",
    "zca_catalog_a0.25",
    "zca_catalog_a0.5",
    "zca_catalog_a0.75",
    "zca_catalog_a1.0",
    "zca_catalog_books",
    "zca_catalog_grocery",
]
CATALOG_TESTED = [
    c for c in CATALOG_ALL if c != "pca_catalog"
]  # PCA is a control outside the tests
ROWS_MAIN = [
    ("Non-personalised", ["random", "popularity", "top_category"]),
    ("No correction", ["raw"]),
    (
        "Catalog side",
        ["center_catalog", "abtt_catalog_D10", "zca_catalog_a1.0"] + ["zca_catalog_books"],
    ),
    ("Query side", ["zca_profile"]),
    ("Both sides", ["zca_both_separate", "zca_symmetric"]),
]
ROWS_FULL = [
    ("Non-personalised", ["random", "popularity", "top_category"]),
    ("No correction", ["raw"]),
    ("Catalog side", CATALOG_ALL),
    ("Query side", ["zca_profile"]),
    ("Both sides", ["zca_both_separate", "zca_symmetric"]),
]
BASELINES = ("random", "popularity", "top_category")
SIGNAL_LABEL = {"zca_catalog_a1.0": "Catalog ZCA", "zca_profile": "Query ZCA"}
GROUP_PREFIX = {
    "Non-personalised": "",
    "No correction": "",
    "Catalog side": "Catalog: ",
    "Query side": "Query: ",
    "Both sides": "Both: ",
}


# ───────────────────────────── formatting helpers ──────────────────────────────


def tex(s: str) -> str:
    """Escape LaTeX specials in a plain-text label."""
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def fnum(x: float, nd: int) -> str:
    return "--" if x is None or not np.isfinite(x) else f"{x:.{nd}f}"


def signed(x: float, nd: int = 0, suffix: str = "") -> str:
    """Signed number with a math minus, e.g. ``$-$40``."""
    if x is None or not np.isfinite(x):
        return "--"
    if round(abs(x), nd) == 0:
        return f"0{suffix}"
    sign = "$-$" if x < 0 else "$+$"
    return f"{sign}{abs(x):.{nd}f}{suffix}"


def small(x: float, nd: int = 3) -> str:
    """Fixed decimals; values that would print as zero show ``0`` or ``~0``."""
    if x is None or not np.isfinite(x):
        return "--"
    if abs(x) < 0.5 * 10.0**-nd:
        return "0" if x == 0 else r"$\approx 0$"
    return f"{x:.{nd}f}"


def pval(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "--"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def ptex(p: float) -> str:
    """``$p<0.001$`` or ``$p=0.081$``."""
    v = pval(p)
    return f"$p{v}$" if v.startswith("<") else f"$p={v}$"


def dci(d: float, lo: float, hi: float, scale: float = 1e3, nd: int = 2) -> str:
    """Mean difference with its 95% interval, scaled (default units of 1e-3)."""
    return f"{signed(d * scale, nd)} [{signed(lo * scale, nd)}, {signed(hi * scale, nd)}]"


def thousands(n: int) -> str:
    """Integer with a math-safe thousands separator, e.g. ``4{,}884``."""
    return f"{n:,}".replace(",", "{,}")


def mark(p: float, delta: float) -> str:
    """Up/down arrow when ``p`` (Holm) < 0.05, by the sign of ``delta``."""
    if p is None or not np.isfinite(p) or p >= ALPHA:
        return ""
    return UP if delta > 0 else DOWN


def bold(s: str, on: bool) -> str:
    return rf"\textbf{{{s}}}" if on else s


def wrap(body: str, cols: str) -> str:
    """Compact full-width ``tabular*`` (stretched column gaps, uniform font) around ``body``."""
    return (
        "\\begingroup\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
        f"\\begin{{tabular*}}{{\\textwidth}}{{@{{\\extracolsep{{\\fill}}}}{cols}}}\n"
        f"\\toprule\n{body}\\bottomrule\n\\end{{tabular*}}\n\\endgroup\n"
    )


# ─────────────────────────────────── loading ───────────────────────────────────


class Results:
    def __init__(self, tag: str = "") -> None:
        """``tag="_source2"`` loads the files of the movies-as-source pairs (only the families
        that were run for them); the default loads the books-source files."""
        rd = lambda n: pd.read_csv(GRAN / n.replace(".csv", f"{tag}.csv"))  # noqa: E731
        self.summary = rd("summary_test.csv").set_index(
            ["domain", "encoder", "text", "granularity", "condition"]
        )
        self.gap = rd("stats_gap.csv")
        self.vs_pop = rd("stats_vs_popularity.csv")
        self.vs_top = rd("stats_vs_top_category.csv")
        self.vs_raw = rd("stats_vs_raw.csv")
        self.qn = rd("summary_query_norm.csv").set_index(["domain", "encoder", "text", "condition"])
        self.qn_stats = rd("stats_query_norm.csv")
        self.qn_tally = rd("tally_query_norm.csv")
        self.lex = rd("summary_lexical.csv").set_index(["domain", "text", "condition"])
        self.lex_stats = rd("stats_lexical.csv")
        self.knn = rd("summary_knn.csv").set_index(["domain", "text", "condition"])
        self.knn_stats = rd("stats_knn.csv")
        self.br_stats = rd("stats_bridge.csv")
        self.br = rd("summary_bridge.csv").set_index(["domain", "text", "condition"])
        if tag:
            self.qn_val = rd("stats_query_norm_val.csv")
            return
        self.ctl = rd("summary_controls.csv").set_index(["domain", "text", "condition"])
        self.ctl_stats = rd("stats_controls.csv")
        self.hyb_stats = rd("stats_hybrid.csv")
        self.qn_val = rd("stats_query_norm_val.csv")
        self.ms = rd("multisplit_summary.csv")
        self.ms_ps = rd("multisplit_per_split.csv")
        self.ms_share = rd("multisplit_centring_share.csv")
        self.lo_stats = rd("stats_lexical_overlap.csv")
        self.lo_sum = rd("summary_lexical_overlap.csv")
        self.lo = {d: json.loads((GRAN / f"lexical_overlap_{d}.json").read_text()) for d in DOMAINS}

    def row(self, domain, enc, text, gran, cond) -> pd.Series:
        return self.summary.loc[(domain, enc, text, gran, cond)]

    def rel(self, domain, enc, text, gran, cond) -> tuple[float, float]:
        """Relative nDCG@10 gain over raw (fraction) and its Holm p-value."""
        r = self.row(domain, enc, text, gran, cond)
        raw = self.row(domain, enc, text, gran, "raw")["ndcg10"]
        return float(r["ndcg10_delta"] / raw), float(r["ndcg10_p_holm_signflip"])

    def gapped(self, domain, enc, text, gran, cond) -> pd.Series:
        g = self.gap
        sel = (
            (g["domain"] == domain)
            & (g["encoder"] == enc)
            & (g["text"] == text)
            & (g["gt"] == gran)
            & (g["metric"] == "ndcg10")
            & (g["condition"] == cond)
        )
        return g[sel].iloc[0]

    def baseline_test(self, which, domain, enc, text, gran, cond) -> pd.Series:
        t = self.vs_pop if which == "popularity" else self.vs_top
        sel = (
            (t["domain"] == domain)
            & (t["encoder"] == enc)
            & (t["text"] == text)
            & (t["gt"] == gran)
            & (t["metric"] == "ndcg10")
            & (t["condition"] == cond)
        )
        return t[sel].iloc[0]


def run_json(domain: str, enc: str = "gte", text: str = "titlecat") -> dict:
    return json.loads((GRAN / f"{domain}_{enc}_{text}.json").read_text())


def mech_json(domain: str) -> dict:
    return json.loads((MECH / f"{domain}_gte.json").read_text())


def levels_of(domain: str) -> list[str]:
    return run_json(domain)["granularities"]


# ──────────────────────────────── Table 1: data ────────────────────────────────


def table_domains() -> str:
    stats = {
        d: json.loads((ROOT / "data" / "splits" / f"{d}_stats.json").read_text()) for d in DOMAINS
    }
    prep = {
        "music": ROOT / "data/raw/preprocessing_stats.json",
        "movies": ROOT / "data/raw/movies/preprocessing_stats.json",
        "games": ROOT / "data/raw/games/preprocessing_stats.json",
    }
    prep = {d: json.loads(p.read_text()) for d, p in prep.items()}
    runs = {d: run_json(d) for d in DOMAINS}
    mech = {d: mech_json(d) for d in DOMAINS}

    def three(f) -> str:
        return " & ".join(f(d) for d in DOMAINS)

    a = "Quantity & " + " & ".join(DNAME[d] for d in DOMAINS) + " \\\\\n\\midrule\n"
    a += (
        "Target catalog (items) & "
        + three(lambda d: f"{mech[d]['A']['catalog']['raw']['n']:,}")
        + " \\\\\n"
    )
    a += "Eligible users & " + three(lambda d: f"{stats[d]['n_eligible']:,}") + " \\\\\n"
    a += (
        "Train / validation / test users & "
        + three(
            lambda d: f"{stats[d]['n_train']:,} / {stats[d]['n_val']:,} / {stats[d]['n_test']:,}"
        )
        + " \\\\\n"
    )
    a += (
        "Test users evaluated & "
        + three(lambda d: f"{runs[d]['n_eval_users']['test']:,}")
        + " \\\\\n"
    )
    a += (
        "Mean books per user (all history) & "
        + three(lambda d: fnum(prep[d]["book_history"]["mean"], 1))
        + " \\\\\n"
    )
    a += (
        "Mean books pooled in the profile (test) & "
        + three(lambda d: fnum(runs[d]["mean_books_pooled_eval"]["test"], 1))
        + " \\\\\n"
    )
    a += (
        "Mean target items per user (exact relevance) & "
        + three(lambda d: fnum(prep[d]["ground_truth"]["mean"], 1))
        + " \\\\\n"
    )
    tax = json.loads((GRAN / "taxonomy.json").read_text())["domains"]
    a += (
        "Items whose category path does not start with the department (\\%) & "
        + three(lambda d: fnum(100 * (1 - tax[d]["position0"]["share_modal"]), 1))
        + " \\\\\n"
    )
    panel_a = wrap(a, "lrrr")

    b = (
        "Domain & Level & Categories & Coverage & Median relevant set & Random & Top category \\\\\n"
        " & & & (\\% items) & (\\% of catalog) & nDCG@10 & (constant list) \\\\\n\\midrule\n"
    )
    for i, d in enumerate(DOMAINS):
        run = runs[d]
        depths = run["category_depths"]
        levels = run["granularities"]
        first = True
        for lvl in levels:
            rel = run["relevance_size"]["test"][lvl]["median_rel_pct_of_catalog"]
            rnd = run["means"]["test"][lvl]["random"]["nDCG@10"]
            if lvl == "exact":
                cats, cov, top = "--", "--", run["top_category"]["d2"] + r"$^{b}$"
            else:
                cats = f"{depths[lvl]['n_categories']:,}"
                cov = fnum(100 * depths[lvl]["coverage"], 1)
                top = tex(run["top_category"][lvl])
            b += (
                f"{DNAME[d] if first else ''} & {lvl} & {cats} & {cov} & {small(rel, 3) if rel < 0.1 else fnum(rel, 1)} & "
                f"{fnum(rnd, 4) if lvl == 'exact' else fnum(rnd, 3)} & {top} \\\\\n"
            )
            first = False
        skipped = [k for k, v in depths.items() if not v["evaluated"]]
        if skipped:
            sk = ", ".join(f"{k} ({fnum(100 * depths[k]['coverage'], 0)}\\%)" for k in skipped)
            b += f" & \\multicolumn{{6}}{{l}}{{\\textit{{not evaluated (coverage $<50\\%$): {sk}}}}} \\\\\n"
        if i < len(DOMAINS) - 1:
            b += "\\midrule\n"
    panel_b = wrap(b, "llrrrrl")
    return panel_a + "\\par\\medskip\n" + panel_b


# ───────────────────────── Tables 2 and appendix: conditions ───────────────────


def _col_specs(domain: str, full: bool) -> list[tuple[str, str, str, int]]:
    """(granularity, metric key, header, decimals) of the numeric columns."""
    cols = [("exact", "ndcg10", "exact nDCG@10", 4), ("exact", "r100", "exact R@100", 4)]
    levels = [g for g in levels_of(domain) if g != "exact"]
    if not full:
        levels = [g for g in levels if g in ("d2", "d1")]
    cols += [(g, "ndcg10", f"{g} nDCG@10", 3) for g in sorted(levels, reverse=True)]
    return cols


def table_conditions(
    R: Results, domains, rows, full: bool, enc="gte", text="titlecat", inline: bool = False
) -> str:
    ncol = max(len(_col_specs(d, full)) for d in domains)
    body = ""
    for di, d in enumerate(domains):
        cols = _col_specs(d, full)
        n = int(R.row(d, enc, text, "exact", "raw")["n"])
        hdr = " & ".join(h for _, _, h, _ in cols)
        if di:
            body += "\\midrule\n"
        body += f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\textbf{{{DNAME[d]}}} ($n={thousands(n)}$ test users)}} \\\\\n"
        body += f"\\cmidrule(lr){{2-{ncol + 1}}}\nCondition & {hdr} \\\\\n\\midrule\n"
        allc = [c for _, cs in rows for c in cs]
        best = {
            (g, m): max(float(R.row(d, enc, text, g, c)[m]) for c in allc) for g, m, _, _ in cols
        }
        for gi, (grp, conds) in enumerate(rows):
            if gi:
                body += "\\addlinespace\n"
            prefix = ""
            if inline:  # no heading rows: the group is a label prefix
                prefix = GROUP_PREFIX[grp]
            elif len(rows) > 1 and grp not in ("No correction",):
                body += f"\\multicolumn{{{ncol + 1}}}{{l}}{{\\textit{{{grp}}}}} \\\\\n"
            for c in conds:
                cells = []
                for g, m, _, nd in cols:
                    r = R.row(d, enc, text, g, c)
                    v = float(r[m])
                    mk = ""
                    if c not in BASELINES and c != "raw":
                        mk = mark(r[f"{m}_p_holm_signflip"], r[f"{m}_delta"])
                    cells.append(bold(fnum(v, nd), v >= best[(g, m)] - 1e-12) + mk)
                cells += [""] * (ncol - len(cells))
                lead = prefix if inline else "\\quad "
                body += f"{lead}{COND[c]} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "r" * ncol)


ROWS_COMBINED = [
    ("Non-personalised", ["random", "popularity", "top_category"]),
    ("No correction", ["raw"]),
    ("Catalog side", ["center_catalog", "abtt_catalog_D10", "pca_catalog", "zca_catalog_a1.0", "zca_catalog_books"]),
    ("Query side", ["zca_profile"]),
    ("Both sides", ["zca_both_separate", "zca_symmetric"]),
]  # fmt: skip


def table_full(R: Results, enc="gte", text="titlecat") -> str:
    """Compact grid: exact and d2 nDCG@10 per domain (full grids: results/granularity/summary_*.csv)."""
    cols = [(d, g) for d in DOMAINS for g in ("exact", "d2")]
    allc = [c for _, cs in ROWS_COMBINED for c in cs]
    best = {k: max(float(R.row(*k[:1], enc, text, k[1], c)["ndcg10"]) for c in allc) for k in cols}
    n = len(cols)
    body = (
        "Condition & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n"
    )
    body += "".join(f"\\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DOMAINS)))
    body += "\n & " + " & ".join("exact & $d_2$" for _ in DOMAINS) + " \\\\\n\\midrule\n"
    for gi, (grp, conds) in enumerate(ROWS_COMBINED):
        if gi:
            body += "\\addlinespace\n"
        for c in conds:
            cells = []
            for d, g in cols:
                r = R.row(d, enc, text, g, c)
                v = float(r["ndcg10"])
                mk = ""
                if c not in BASELINES and c != "raw":
                    mk = mark(r["ndcg10_p_holm_signflip"], r["ndcg10_delta"])
                cells.append(
                    bold(fnum(v, 4 if g == "exact" else 3), v >= best[(d, g)] - 1e-12) + mk
                )
            body += f"{GROUP_PREFIX[grp]}{COND[c]} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "r" * n)


# ───────────────────────────── Table 3: music encoders ─────────────────────────


def _delta_cell(R, enc, gran, cond, domain="music", text="titlecat") -> tuple[str, float, float]:
    g, p = R.rel(domain, enc, text, gran, cond)
    mk = STAR if np.isfinite(p) and p < ALPHA else ""
    return signed(100 * g, 0) + mk, g, p


def _gap_cell(R, enc, gran, cond, domain="music", text="titlecat") -> tuple[str, int]:
    g = R.gapped(domain, enc, text, gran, cond)
    sig = int(g["ci95_low"] > 0) - int(g["ci95_high"] < 0)
    return signed(100 * g["gap"], 0) + (DAGGER if sig else ""), int(sig)


def table_encoders_gap(R: Results) -> str:
    """Music encoders: relative gain of query ZCA and catalog ZCA at the exact level and d2, and the
    granularity gap; the sign counts are given for the verified encoders first."""
    sides = (("zca_profile", "Query ZCA"), ("zca_catalog_a1.0", r"Catalog ZCA ($\alpha{=}1$)"))
    body = (
        " & & & & \\multicolumn{3}{c}{"
        + sides[0][1]
        + r"} & \multicolumn{3}{c}{"
        + sides[1][1]
        + "} \\\\\n"
        "\\cmidrule(lr){5-7}\\cmidrule(lr){8-10}\n"
        "Encoder & Prov. & Text & Raw exact & exact & d2 & gap & exact & d2 & gap \\\\\n\\midrule\n"
    )
    counts = {
        k: {"verified": [0, 0, 0], "all": [0, 0, 0]} for k in ("pe", "pd", "pg", "ce", "cd", "cg")
    }

    def tally(key, enc, s):  # s: +1 / 0 / -1
        idx = 0 if s > 0 else 2 if s < 0 else 1
        counts[key]["all"][idx] += 1
        if ENCODERS[enc][1]:
            counts[key]["verified"][idx] += 1

    for enc, (name, ok, txt) in ENCODERS.items():
        raw = float(R.row("music", enc, "titlecat", "exact", "raw")["ndcg10"])
        cells = []
        for (cond, _), (ke, kd, kg) in zip(sides, (("pe", "pd", "pg"), ("ce", "cd", "cg"))):
            ce, g, p = _delta_cell(R, enc, "exact", cond)
            tally(ke, enc, 0 if not p < ALPHA else (1 if g > 0 else -1))
            cd, g, p = _delta_cell(R, enc, "d2", cond)
            tally(kd, enc, 0 if not p < ALPHA else (1 if g > 0 else -1))
            cg, sg = _gap_cell(R, enc, "d2", cond)
            tally(kg, enc, sg)
            cells += [ce, cd, cg]
        body += (
            f"{name} & {'V' if ok else 'U'} & {txt} & {fnum(raw, 4)} & "
            + " & ".join(cells)
            + " \\\\\n"
        )
    body += "\\midrule\n"
    for label, which in (("Verified (8)", "verified"), ("All 10 encoders", "all")):
        cells = [
            " / ".join(str(v) for v in counts[k][which])
            for k in ("pe", "pd", "pg", "ce", "cd", "cg")
        ]
        body += f"{label}: sign counts (+/ns/$-$) & & & & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "lllrrrrrrr")


# ───────────────────────────── Table 4: mechanism ──────────────────────────────


def table_mechanism() -> str:
    """Diagnostics A, B, C and E side by side: one column quadruple (raw, catalog ZCA, centred
    query, query ZCA) per domain; a dash where a diagnostic does not apply to the condition."""
    mech = {d: mech_json(d) for d in DOMAINS}
    cond = {"raw": "raw", "cat": "zca_catalog_a1.0", "ctr": "center_profile", "prof": "zca_profile"}

    def row(label, get, nd, fn=fnum):
        cells = [get(mech[d], w) for d in DOMAINS for w in cond]
        out = " & ".join("--" if v is None else fn(v, nd) for v in cells)
        return f"\\quad {label} & {out} \\\\\n"

    def at(sec, part, key, sides):
        return lambda m, w: m[sec][part][cond[w]][key] if w in sides else None

    def section(title):
        return f"\\multicolumn{{13}}{{l}}{{\\textit{{{title}}}}} \\\\\n"

    head = " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS) + " \\\\\n"
    head += "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3)) + "\n"
    head += " & " + " & ".join(["Raw", "Cat.", "Ctr.", "Prof."] * 3) + " \\\\\n\\midrule\n"
    body = head + section("(A) Collapse")
    for label, part, key, nd, fn, sides in (
        (
            "Profiles: mean pairwise cosine",
            "profiles",
            "mean_pairwise_cosine",
            3,
            small,
            ("raw", "prof"),
        ),
        ("Profiles: effective rank", "profiles", "effective_rank", 0, fnum, ("raw", "prof")),
        (
            "Items: mean pairwise cosine",
            "catalog",
            "mean_pairwise_cosine",
            3,
            small,
            ("raw", "cat"),
        ),
        ("Items: effective rank", "catalog", "effective_rank", 0, fnum, ("raw", "cat")),
    ):
        body += row(label, at("A", part, key, sides), nd, fn)
    body += section("(B) Top-10 lists")
    for label, key, nd, fn in (
        ("Catalog coverage (\\%)", "coverage_pct", 1, fnum),
        ("Exposure Gini", "gini_exposure", 3, fnum),
        ("Overlap between users (Jaccard)", "mean_jaccard", 3, small),
        ("Hubness skewness", "hubness_skewness_k10", 1, fnum),
        ("Mean popularity percentile of the items", "mean_popularity_percentile", 1, fnum),
    ):
        body += row(label, at("B", "conditions", key, ("raw", "cat", "ctr", "prof")), nd, fn)
    body += section("(C) Category structure of the items")
    for label, key, nd in (
        ("Between/within variance ratio", "between_within_ratio", 3),
        ("Same-category share of 10 nearest items", "nn10_same_category", 2),
    ):
        body += row(label, at("C", "conditions", key, ("raw", "cat")), nd)
    body += section("(E) Category concentration of the top-10 lists")
    for label, key, nd in (
        ("Entropy, depth-2 categories (bits)", "entropy_bits", 2),
        ("Share in the 5 most frequent train categories", "share_top5_train_categories", 2),
    ):
        body += row(label, at("E", "concentration", key, ("raw", "cat", "ctr", "prof")), nd)
    return wrap(body, "l" + "rrrr" * 3)


def table_mechanism_full() -> str:
    """Category concentration (E) and list structure (B) for every condition."""
    conds = (
        "raw",
        "zca_catalog_a1.0",
        "zca_profile",
        "zca_both_separate",
        "center_profile",
        "popularity",
        "top_category",
    )
    head = "Condition & Entropy & Categories & Top-1 share & Top-5 share & Jaccard to top-category list & $\\rho$ (user, d2 nDCG@10) \\\\\n\\midrule\n"
    body = head
    for di, d in enumerate(DOMAINS):
        m = mech_json(d)["E"]
        if di:
            body += "\\midrule\n"
        body += f"\\multicolumn{{7}}{{l}}{{\\textbf{{{DNAME[d]}}} ({m['n_categories_catalog']} depth-2 categories in the catalog)}} \\\\\n"
        for c in conds:
            e = m["concentration"][c]
            j = m["jaccard_to_top_category"][c]
            rho = j["spearman_vs_d2_ndcg10"]["rho"]
            body += (
                f"\\quad {COND[c]} & {fnum(e['entropy_bits'], 2)} & {e['n_distinct_categories']} & "
                f"{fnum(e['share_top1_train_category'], 3)} & {fnum(e['share_top5_train_categories'], 3)} & "
                f"{small(j['mean_jaccard'], 4)} & {fnum(rho, 3) if rho is not None else '--'} \\\\\n"
            )
        pooled = m["jaccard_to_top_category"]["pooled_personalised"]["spearman_vs_d2_ndcg10"]
        body += (
            f"\\quad Pooled personalised conditions & \\multicolumn{{5}}{{r}}{{}} & {fnum(pooled['rho'], 3)} "
            f"({ptex(pooled['p'])}) \\\\\n"
        )
    return wrap(body, "lrrrrrr")


# ───────────────────── Signal, text format, baselines, alphas ──────────────────


def table_signal() -> str:
    """Users with at least one top-10 hit (exact relevance) and typicality link (D)."""
    conds = ("raw", "zca_catalog_a1.0", "zca_profile", "popularity", "top_category")
    head = (
        "Domain & Test users & "
        + " & ".join(SIGNAL_LABEL.get(c, COND[c]) for c in conds)
        + " & Raw or profile ZCA"
        " & $\\rho$ (all) & $\\rho$ (users with a hit) \\\\\n\\midrule\n"
    )
    body = head
    for d in DOMAINS:
        df = pd.read_csv(
            GRAN / f"{d}_gte_titlecat.csv.gz",
            usecols=["user_id", "split", "gt", "condition", "nDCG@10"],
        )
        df = df[(df["split"] == "test") & (df["gt"] == "exact")]
        wide = df.pivot(index="user_id", columns="condition", values="nDCG@10")
        n = len(wide)
        hit = wide > 0
        union = int((hit["raw"] | hit["zca_profile"]).sum())
        D = mech_json(d)["D"]
        assert union == D["n_users_nonzero_ndcg"], (d, union, D["n_users_nonzero_ndcg"])
        cells = [f"{int(hit[c].sum()):,} ({100 * hit[c].mean():.1f}\\%)" for c in conds]
        body += (
            f"{DNAME[d]} & {n:,} & "
            + " & ".join(cells)
            + f" & {union:,} ({100 * union / n:.1f}\\%) & "
            f"{fnum(D['spearman_rho'], 3)} ({ptex(D['p'])}) & "
            f"{fnum(D['spearman_rho_nonzero'], 3)} ({ptex(D['p_nonzero'])}) \\\\\n"
        )
    return wrap(body, "lrrrrrrrrr")


def table_baselines(R: Results) -> str:
    """Best correction per level against the two constant lists (paired, Holm, sign-flip test)."""
    body = (
        "Domain & Level & Popularity & Top-category & Best method & nDCG@10 & "
        "$\\Delta$ vs popularity [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ & "
        "$\\Delta$ vs top-category [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
    )
    for di, d in enumerate(DOMAINS):
        if di:
            body += "\\midrule\n"
        for gi, g in enumerate(levels_of(d)):
            sub = R.summary.loc[(d, "gte", "titlecat", g)]
            pool = [c for c in sub.index if c not in BASELINES]
            best = max(pool, key=lambda c: sub.loc[c, "ndcg10"])
            nd = 4 if g == "exact" else 3
            sc, dd = (1e3, 2) if g == "exact" else (1, 3)
            cells = []
            for which in ("popularity", "top_category"):
                t = R.baseline_test(which, d, "gte", "titlecat", g, best)
                cells.append(dci(t["mean_diff"], t["ci95_low"], t["ci95_high"], sc, dd))
                cells.append(f"{int(t['n_nonzero_diff']):,}")
                cells.append(pval(t["p_holm_signflip"]))
            body += (
                f"{DNAME[d] if gi == 0 else ''} & {g} & {fnum(sub.loc['popularity', 'ndcg10'], nd)} & "
                f"{fnum(sub.loc['top_category', 'ndcg10'], nd)} & {COND[best]} & {fnum(sub.loc[best, 'ndcg10'], nd)} & "
                + " & ".join(cells)
                + " \\\\\n"
            )
    return wrap(body, "llrrlrrrrrrr")


def provenance_ok(domain: str, enc: str) -> bool:
    """Verified embeddings (docs/EMBEDDINGS.md): everything but qwen3 outside music and minilm-l12."""
    return ENCODERS[enc][1] and not (enc == "qwen3" and domain != "music")


def text_code(domain: str, enc: str, text: str) -> str:
    """TC, LT, ? as in ENCODERS; ``TC+D`` adds the description, ``T`` is the title only; BGE-large
    pairs TC items with LT books outside music; Qwen3 outside music cannot be recovered."""
    if text == "desc":
        return "TC+D"
    if text == "title":
        return "T"
    if enc == "bge-large" and domain != "music":
        return "TC/LT$^{b}$"
    if enc == "qwen3" and domain != "music":
        return "?"
    return ENCODERS[enc][2]


def catalog_counts(R: Results, d: str, e: str, t: str) -> tuple[int, int, int]:
    """Catalog-side corrections of a run (exact relevance): Holm-significantly better / not
    significant / significantly worse than raw, over the conditions of CATALOG_ALL present."""
    v = R.vs_raw
    sel = (
        (v["domain"] == d)
        & (v["encoder"] == e)
        & (v["text"] == t)
        & (v["gt"] == "exact")
        & (v["metric"] == "ndcg10")
        & v["condition"].isin(CATALOG_TESTED)
    )
    x = v[sel]
    sig = x["p_holm_signflip"] < ALPHA
    pos, neg = int((sig & (x["mean_diff"] > 0)).sum()), int((sig & (x["mean_diff"] < 0)).sum())
    return pos, len(x) - pos - neg, neg


def best_catalog(R: Results, d: str, e: str, t: str) -> str:
    """Largest relative exact-level gain among the catalog-side corrections of a run, with the
    Holm star."""
    sub = R.summary.loc[(d, e, t, "exact")]
    rows = [(c, *R.rel(d, e, t, "exact", c)) for c in CATALOG_ALL if c in sub.index]
    _, g, p = max(rows, key=lambda r: r[1])
    return signed(100 * g, 0) + (STAR if p < ALPHA else "")


def table_encoders_all(R: Results) -> str:
    """Every run: relative gains of the two single-side ZCA conditions at each level, and the
    outcome of all catalog-side corrections at exact relevance."""
    body = (
        " & & & & & \\multicolumn{3}{c}{Query ZCA} & \\multicolumn{3}{c}{Catalog ZCA ($\\alpha{=}1$)} & \\multicolumn{2}{c}{All catalog-side, exact} \\\\\n"
        "\\cmidrule(lr){6-8}\\cmidrule(lr){9-11}\\cmidrule(lr){12-13}\n"
        "Domain & Encoder & Prov. & Text & Raw exact & exact & d2 & d1 & exact & d2 & d1 & best & $+$/ns/$-$ \\\\\n\\midrule\n"
    )
    runs = sorted(
        {(d, e, t) for d, e, t, _, _ in R.summary.index},
        key=lambda x: (DOMAINS.index(x[0]), list(ENCODERS).index(x[1]), x[2] != "titlecat"),
    )
    last = None
    for d, e, t in runs:
        if last is not None and last != d:
            body += "\\midrule\n"
        cells = [
            _delta_cell(R, e, g, c, d, t)[0]
            for c in ("zca_profile", "zca_catalog_a1.0")
            for g in ("exact", "d2", "d1")
        ]
        cells.append(best_catalog(R, d, e, t))
        cells.append(" / ".join(str(v) for v in catalog_counts(R, d, e, t)))
        raw = float(R.row(d, e, t, "exact", "raw")["ndcg10"])
        tname = text_code(d, e, t)
        prov = "V" if provenance_ok(d, e) else "U"
        body += (
            f"{DNAME[d] if last != d else ''} & {ENC_LABEL[e]} & {prov} & {tname} & {fnum(raw, 4)} & "
            + " & ".join(cells)
            + " \\\\\n"
        )
        last = d
    return wrap(body, "llllrrrrrrrrr")


# ─────────────────────── Hybrid (content + popularity) tables ───────────────────────

HYBRID_ROWS = (  # (condition in hybrid_{domain}_gte.csv.gz, label, hybrid of a pure condition)
    ("popularity", "Popularity list", None),
    ("top_category", "Top-category list", None),
    ("raw", "Raw", None),
    ("hybrid_raw", "Raw + popularity", "raw"),
    ("zca_profile", "Query ZCA", None),
    ("hybrid_zca_profile", "Query ZCA + popularity", "zca_profile"),
    ("zca_catalog_a1.0", "Catalog ZCA", None),
    ("hybrid_zca_catalog_a1.0", "Catalog ZCA + popularity", "zca_catalog_a1.0"),
)


def hybrid_json(domain: str) -> dict:
    return json.loads((GRAN / f"hybrid_{domain}_gte.json").read_text())


def lam_tex(label: str) -> str:
    return r"$\infty$" if label == "inf" else label


def table_hybrid() -> str:
    """Content + lambda * popularity: TEST means at the lambda chosen on VAL, and the paired
    differences of the hybrids against the popularity list (Holm within domain and metric,
    sign-flip test on the mean)."""
    st = pd.read_csv(GRAN / "stats_hybrid.csv")
    body = (
        "Domain & Method & $\\lambda$ & nDCG@10 & R@100 & $d_2$ nDCG@10 & "
        "$\\Delta$ nDCG@10 vs popularity [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ & "
        "$\\Delta$ R@100 vs popularity [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
    )
    for di, d in enumerate(DOMAINS):
        if di:
            body += "\\midrule\n"
        j = hybrid_json(d)
        m = j["test"]
        col = {  # best value of each column in the domain
            (g, k): max(m[g][c][k] for c, _, _ in HYBRID_ROWS)
            for g, k in (("exact", "nDCG@10"), ("exact", "R@100"), ("d2", "nDCG@10"))
        }
        for ri, (cond, label, pure) in enumerate(HYBRID_ROWS):
            cells = [
                bold(fnum(m[g][cond][k], nd), m[g][cond][k] == col[(g, k)])
                for g, k, nd in (
                    ("exact", "nDCG@10", 4),
                    ("exact", "R@100", 4),
                    ("d2", "nDCG@10", 3),
                )
            ]
            lam = lam_tex(j["selected_lambda"][pure]) if pure else "--"
            tests = []
            for metric in ("ndcg10", "r100"):
                sel = st[
                    (st["domain"] == d)
                    & (st["metric"] == metric)
                    & (st["condition"] == cond)
                    & (st["reference"] == "popularity")
                ]
                if sel.empty:
                    tests += ["--", "--", "--"]
                    continue
                t = sel.iloc[0]
                tests.append(dci(t["mean_diff"], t["ci95_low"], t["ci95_high"]))
                tests.append(f"{int(t['n_nonzero_diff']):,}")
                tests.append(pval(t["p_holm_signflip"]))
            body += (
                f"{DNAME[d] if ri == 0 else ''} & {label} & {lam} & "
                + " & ".join(cells + tests)
                + " \\\\\n"
            )
    return wrap(body, "llrrrrrrrrrr")


def table_hybrid_val() -> str:
    """VAL exact nDCG@10 (x1000) of content + lambda * popularity for the whole lambda grid;
    the lambda chosen on it is in bold."""
    lams = hybrid_json("music")["lambdas"]
    body = (
        "Domain & Content & "
        + " & ".join(f"$\\lambda{{=}}{lam_tex(x)}$" if x != "inf" else lam_tex(x) for x in lams)
        + " \\\\\n\\midrule\n"
    )
    for di, d in enumerate(DOMAINS):
        if di:
            body += "\\midrule\n"
        j = hybrid_json(d)
        for ci, (c, label) in enumerate(
            (("raw", "Raw"), ("zca_profile", "Query ZCA"), ("zca_catalog_a1.0", "Catalog ZCA"))
        ):
            curve = j["val_curve"][c]
            cells = [
                bold(fnum(1e3 * curve[x]["nDCG@10"], 2), x == j["selected_lambda"][c]) for x in lams
            ]
            body += f"{DNAME[d] if ci == 0 else ''} & {label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "ll" + "r" * len(lams))


# ─────────────────── Query normalisation: side x source x operation ────────────────

GTE, TC = "gte", "titlecat"
SRC_LAB = {"train_users": "users", "books_catalog": "books", "target_catalog": "target"}
QN_OPS = (  # (row label, query-side arm pattern, item-side arm of the grid)
    ("Centring", "center_{s}", "center_catalog"),
    (r"ABTT ($D{=}1$)", "abtt_{s}_D1", "abtt_catalog_D1"),
    (r"ABTT ($D{=}3$)", "abtt_{s}_D3", "abtt_catalog_D3"),
    (r"ABTT ($D{=}10$)", "abtt_{s}_D10", "abtt_catalog_D10"),
    (r"ZCA ($\alpha{=}0.25$)", "zca_{s}_a0.25", "zca_catalog_a0.25"),
    (r"ZCA ($\alpha{=}0.5$)", "zca_{s}_a0.5", "zca_catalog_a0.5"),
    (r"ZCA ($\alpha{=}0.75$)", "zca_{s}_a0.75", "zca_catalog_a0.75"),
    (r"ZCA ($\alpha{=}1$)", "zca_{s}_a1.0", "zca_catalog_a1.0"),
    (r"ZCA, shrunk $\Sigma$", "zcalw_{s}", None),
)
COND["zca_catalog_train_users"] = r"ZCA, $\Sigma$ of users"
QN_CONTRASTS = (  # (family, test name in stats_query_norm.csv, label)
    ("contrast", "a_zca_train_users_vs_zca_catalog", r"(a1) ZCA query (users) vs ZCA catalog"),
    ("contrast", "a_center_train_users_vs_zca_catalog", r"(a2) centring query (users) vs ZCA catalog"),
    ("contrast", "a_zca_books_catalog_vs_zca_catalog", r"(a3) ZCA query (books) vs ZCA catalog"),
    ("contrast", "a_selected_query_vs_selected_catalog", r"(a4) best query arm vs best catalog arm"),
    ("contrast", "b_zca_vs_center_train_users", r"(b) ZCA vs centring, query (users)"),
    ("contrast", "c_both_sides_vs_query_only", r"(c) both sides vs query only"),
    ("post_hoc", "post_hoc:zca_target_catalog_vs_raw", r"(d1)$^{\ddagger}$ ZCA query (target) vs raw"),
    ("post_hoc", "post_hoc:zca_train_users_vs_zca_target_catalog", r"(d2)$^{\ddagger}$ ZCA query (users) vs ZCA query (target)"),
    ("matrix", "matrix:items_U_vs_raw", r"(m1)$^{\S}$ ZCA items (users) vs raw"),
    ("matrix", "matrix:side_U", r"(m2)$^{\S}$ ZCA query vs ZCA items, both with users' statistics"),
    ("matrix", "matrix:side_B", r"(m3)$^{\S}$ ZCA query vs ZCA items, both with books' statistics"),
    ("matrix", "matrix:side_C", r"(m4)$^{\S}$ ZCA query vs ZCA items, both with target statistics"),
)  # fmt: skip
QN_FAMILY_PREFIX = (("post_hoc", "post_hoc"), ("matrix:", "matrix"))


def qn_family(test: str) -> str:
    """Family of a test of stats_query_norm.csv from its name."""
    for prefix, fam in QN_FAMILY_PREFIX:
        if test.startswith(prefix):
            return fam
    return "vs_raw" if test.startswith("vs_raw") else "contrast"


VERIFIED_MUSIC = [e for e, (_, ok, _) in ENCODERS.items() if ok]


def arm_label(c: str) -> str:
    """Short label of a condition of the query-normalisation family."""
    if c in COND:
        return COND[c]
    m = re.fullmatch(
        r"(center|abtt|zca|zcalw)_(train_users|books_catalog|target_catalog)(?:_([aD])([\d.]+))?", c
    )
    assert m, c
    op, src, kind, val = m.groups()
    name = {"center": "centring", "abtt": f"ABTT $D{{=}}{val}$", "zcalw": "shrunk ZCA"}.get(op)
    if op == "zca":
        name = rf"ZCA $\alpha{{=}}{float(val):g}$"
    return f"{name} ({SRC_LAB[src]})"


def _stat_row(df: pd.DataFrame, **kw) -> pd.Series | None:
    sel = np.ones(len(df), bool)
    for k, v in kw.items():
        sel &= (df[k] == v).to_numpy()
    return df[sel].iloc[0] if sel.any() else None


def qn_mark(R: Results, d: str, cond: str, item_side: bool, enc: str = GTE, text: str = TC) -> str:
    """Up/down arrow for an arm against ``raw`` (exact nDCG@10, Holm sign-flip p < 0.05); the
    query-side arms use the family of ``stats_query_norm.csv``, the item-side arms the grid
    family of ``stats_vs_raw.csv``. Arms without a test carry no marker."""
    if item_side:
        r = _stat_row(R.vs_raw, domain=d, encoder=enc, text=text, gt="exact", metric="ndcg10",
                      condition=cond)  # fmt: skip
    else:
        r = _stat_row(R.qn_stats, domain=d, encoder=enc, text=text, metric="ndcg10",
                      family="vs_raw", test=f"vs_raw:{cond}")  # fmt: skip
    return "" if r is None else mark(r["p_holm_signflip"], r["mean_diff"])


MATRIX_KEY = {  # (side, source) -> arm of the query-normalisation family (ZCA, alpha 1)
    ("query", "U"): "zca_train_users_a1.0",
    ("query", "B"): "zca_books_catalog_a1.0",
    ("query", "C"): "zca_target_catalog_a1.0",
    ("items", "U"): "zca_catalog_train_users",
    ("items", "B"): "zca_catalog_books",
    ("items", "C"): "zca_catalog_a1.0",
}
SOURCES3 = ("U", "B", "C")  # training users, book catalog, target catalog


def matrix_test(R: Results, d: str, test: str, enc: str = GTE, text: str = TC) -> pd.Series:
    r = _stat_row(R.qn_stats, domain=d, encoder=enc, text=text, metric="ndcg10", family="matrix",
                  test=test)  # fmt: skip
    assert r is not None, (d, enc, text, test)
    return r


def matrix_mark(R: Results, d: str, side: str, src: str) -> str:
    """Arrow of the cell (side, source) against raw in the Holm family ``matrix``."""
    r = matrix_test(R, d, f"matrix:{side}_{src}_vs_raw")
    return mark(r["p_holm_signflip"], r["mean_diff"])


def table_qnorm(R: Results) -> str:
    """nDCG@10 (x1000, exact relevance) of every operation x statistics source, GTE, per domain:
    the query transformed with the statistics of the training users, of the book catalog or of
    the target catalog, and the items transformed with the target catalog's own statistics."""
    head = " & " + " & ".join(f"\\multicolumn{{4}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS) + " \\\\\n"
    head += "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3)) + "\n"
    head += "Operation" + " & \\multicolumn{3}{c}{Query: statistics of} & Items" * 3 + " \\\\\n"
    head += "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{4 + 4 * i}}}" for i in range(3)) + "\n"
    head += " & users & books & target & target" * 3 + " \\\\\n\\midrule\n"
    allv = {d: [] for d in DOMAINS}
    for _, qp, ip in QN_OPS:
        for d in DOMAINS:
            allv[d] += [qn_row(R, d, qp.format(s=s))["ndcg10"] for s in SRC_LAB]
            if ip:
                allv[d].append(qn_row(R, d, ip)["ndcg10"])
    best = {d: max(v) for d, v in allv.items()}
    body = head
    for cond, label in (("popularity", "Popularity list"), ("raw", "Raw")):
        cells = [
            f"\\multicolumn{{4}}{{c}}{{{fnum(1e3 * qn_row(R, d, cond)['ndcg10'], 2)}}}"
            for d in DOMAINS
        ]
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n"
    for label, qp, ip in QN_OPS:
        cells = []
        for d in DOMAINS:
            for s in SRC_LAB:
                c = qp.format(s=s)
                v = qn_row(R, d, c)["ndcg10"]
                cells.append(bold(fnum(1e3 * v, 2), v >= best[d]) + qn_mark(R, d, c, False))
            if ip:
                v = qn_row(R, d, ip)["ndcg10"]
                cells.append(bold(fnum(1e3 * v, 2), v >= best[d]) + qn_mark(R, d, ip, True))
            else:
                cells.append("--")
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "rrrr" * 3)


def table_matrix(R: Results) -> str:
    """The side x source matrix of ZCA (alpha 1), GTE, exact nDCG@10 x1000: rows are the source of
    the statistics (users U, book catalog B, target catalog C), columns the side that is corrected
    (query or items), per domain, and the matched-source side contrast (query minus items)."""
    head = (
        "Statistics of & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3))
        + "\n & "
        + " & ".join("query & items & query $-$ items" for _ in DOMAINS)
        + " \\\\\n\\midrule\n"
    )
    body = head
    for cond, label in (("popularity", "Popularity list"), ("raw", "Raw (no correction)")):
        cells = [
            f"\\multicolumn{{3}}{{c}}{{{fnum(1e3 * qn_row(R, d, cond)['ndcg10'], 2)}}}"
            for d in DOMAINS
        ]
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n"
    best = {
        d: max(qn_row(R, d, MATRIX_KEY[(side, src)])["ndcg10"] for side in ("query", "items")
               for src in SOURCES3)
        for d in DOMAINS
    }  # fmt: skip
    for src, label in (("U", "users U"), ("B", "book catalog B"), ("C", "target catalog C")):
        cells = []
        for d in DOMAINS:
            for side in ("query", "items"):
                v = qn_row(R, d, MATRIX_KEY[(side, src)])["ndcg10"]
                cells.append(bold(fnum(1e3 * v, 2), v >= best[d]) + matrix_mark(R, d, side, src))
            r = matrix_test(R, d, f"matrix:side_{src}")
            cells.append(
                signed(1e3 * r["mean_diff"], 2) + mark(r["p_holm_signflip"], r["mean_diff"])
            )
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "rrr" * 3)


def qn_row(R: Results, d: str, cond: str, enc: str = GTE, text: str = TC) -> pd.Series:
    return R.qn.loc[(d, enc, text, cond)]


def qn_contrast(R: Results, d: str, enc: str, text: str, test: str) -> pd.Series:
    fam = qn_family(test)
    r = _stat_row(R.qn_stats, domain=d, encoder=enc, text=text, metric="ndcg10", family=fam,
                  test=test)  # fmt: skip
    assert r is not None, (d, enc, text, test)
    return r


def qn_tally(R: Results, test: str, domain: str = "music", text: str = TC) -> str:
    """``better / ns / worse`` over the verified encoders of the first condition against the
    second (Holm sign-flip p < 0.05), from ``tally_query_norm.csv``."""
    t = R.qn_tally
    fam = qn_family(test)
    r = _stat_row(t, domain=domain, text=text, family=fam, test=test, metric="ndcg10",
                  encoders="verified")  # fmt: skip
    return f"{int(r['better'])} / {int(r['ns'])} / {int(r['worse'])}"


def table_qnorm_tests(R: Results) -> str:
    """The paired contrasts of the query-normalisation family (GTE, exact nDCG@10): the six of
    the first family, the two post hoc ones and the four of the side x source matrix."""
    body = (
        "Contrast & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " & Music, 8 encoders \\\\\n"
        + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3))
        + "\\cmidrule(lr){11-11}\n"
        + " & "
        + " & ".join(
            "$\\Delta\\times10^{3}$ [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$"
            for _ in DOMAINS
        )
        + " & better / ns / worse \\\\\n\\midrule\n"
    )
    for _, test, label in QN_CONTRASTS:
        cells = []
        for d in DOMAINS:
            r = qn_contrast(R, d, GTE, TC, test)
            cells += [
                dci(r["mean_diff"], r["ci95_low"], r["ci95_high"]),
                f"{int(r['n_nonzero_diff']):,}",
                pval(r["p_holm_signflip"]),
            ]
        body += f"{label} & " + " & ".join(cells) + f" & {qn_tally(R, test)} \\\\\n"
    return wrap(body, "l" + "rrr" * 3 + "r")


def table_encoders(R: Results) -> str:
    """The eight verified music encoders: nDCG@10 (x1000) of raw, the query normalised with the
    statistics of the users or of the books, with the target catalog's, the catalog whitened, and
    the contrasts of the query-side ZCA with the catalog ZCA (a1), with centring (b) and with
    both sides (c)."""
    body = (
        "Encoder & Text & Raw & \\multicolumn{4}{c}{Query, statistics of} & Items & "
        "\\multicolumn{2}{c}{(a1) vs catalog ZCA} & \\multicolumn{2}{c}{(b) vs centring} & "
        "\\multicolumn{2}{c}{(c) both vs query} \\\\\n"
        "\\cmidrule(lr){4-7}\\cmidrule(lr){9-10}\\cmidrule(lr){11-12}\\cmidrule(lr){13-14}\n"
        " & & & centre & ZCA & ZCA & ZCA & ZCA & $\\Delta\\times10^{3}$ & $p_{\\mathrm{Holm}}$ & "
        "$\\Delta\\times10^{3}$ & $p_{\\mathrm{Holm}}$ & $\\Delta\\times10^{3}$ & $p_{\\mathrm{Holm}}$ \\\\\n"
        " & & & users & users & books & target & target & & & & & & \\\\\n\\midrule\n"
    )
    arms = (  # (arm, item side)
        ("center_train_users", False),
        ("zca_train_users_a1.0", False),
        ("zca_books_catalog_a1.0", False),
        ("zca_target_catalog_a1.0", False),
        ("zca_catalog_a1.0", True),
    )
    for enc in VERIFIED_MUSIC:
        name, _, txt = ENCODERS[enc]
        cells = [fnum(1e3 * qn_row(R, "music", "raw", enc)["ndcg10"], 2)]
        for c, _ in arms:
            v = qn_row(R, "music", c, enc)["ndcg10"]
            cells.append(fnum(1e3 * v, 2) + qn_mark(R, "music", c, False, enc))
        for test in (
            "a_zca_train_users_vs_zca_catalog",
            "b_zca_vs_center_train_users",
            "c_both_sides_vs_query_only",
        ):
            r = qn_contrast(R, "music", enc, TC, test)
            cells += [signed(1e3 * r["mean_diff"], 2), pval(r["p_holm_signflip"])]
        body += f"{name} & {txt} & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n"
    tallies = [
        qn_tally(R, t)
        for t in (
            "a_zca_train_users_vs_zca_catalog",
            "b_zca_vs_center_train_users",
            "c_both_sides_vs_query_only",
        )
    ]
    body += (
        "Verified encoders: better / ns / worse & \\multicolumn{7}{c}{} & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{{t}}}" for t in tallies)
        + " \\\\\n"
    )
    return wrap(body, "llr" + "r" * 5 + "r" * 6)


def tally_counts(df: pd.DataFrame, conds, ver: bool, gt: str) -> tuple[int, int, int, int]:
    """Better / not significant / worse (Holm sign-flip p < 0.05) and the number of tests of the
    ``conds`` of the title+categories runs with verification status ``ver``, exact nDCG@10 or the
    level ``gt``."""
    v = df[(df["text"] == TC) & (df["metric"] == "ndcg10") & (df["gt"] == gt)]
    v = v[v["condition"].isin(conds) & (v["verified"] == ver)]
    sig = v["p_holm_signflip"] < ALPHA
    pos, neg = int((sig & (v["mean_diff"] > 0)).sum()), int((sig & (v["mean_diff"] < 0)).sum())
    return pos, len(v) - pos - neg, neg, len(v)


def table_tally(R: Results) -> str:
    """Sign counts (Holm sign-flip p < 0.05) of the corrections of the grid against raw, the
    popularity list and the top-category list, by side, for verified and unverified encoders.
    Only the title+categories runs; PCA whitening is a control outside the counts."""
    sides = (
        ("Catalog side", CATALOG_TESTED),
        ("Query side (users)", ["zca_profile"]),
        ("Both sides", ["zca_both_separate", "zca_symmetric"]),
    )
    refs = (
        ("raw", R.vs_raw, "exact"),
        ("raw", R.vs_raw, "d2"),
        ("popularity", R.vs_pop, "exact"),
        ("top-category", R.vs_top, "d2"),
    )
    body = (
        "Side & Encoders & Runs & Tests & \\multicolumn{2}{c}{vs raw} & vs popularity & vs top-category \\\\\n"
        "\\cmidrule(lr){5-6}\n & & & & exact & $d_2$ & exact & $d_2$ \\\\\n\\midrule\n"
    )

    for label, conds in sides:
        for ver, vl in ((True, "verified"), (False, "unverified")):
            cells, n_tests = [], None
            for _, df, gt in refs:
                b, ns, w, n = tally_counts(df, conds, ver, gt)
                cells.append(f"{b} / {ns} / {w}")
                n_tests = n
            v = R.vs_raw
            runs = v[(v["text"] == TC) & (v["verified"] == ver)][
                ["domain", "encoder"]
            ].drop_duplicates()
            body += (
                f"{label if ver else ''} & {vl} & {len(runs)} & {n_tests} & "
                + " & ".join(cells)
                + " \\\\\n"
            )
        body += "\\addlinespace\n"
    body += "\\midrule\n" + matrix_tally_rows(R)
    return wrap(body, "llrrrrrr")


# ───────────────────── Lexical, controls, cold and warm items, kNN ─────────────────


def knn_sel(d: str) -> dict[str, int]:
    """Neighbourhood size selected on VAL for each profile variant of the GTE kNN run of ``d``."""
    return json.loads((GRAN / f"knn_{d}_gte.json").read_text())["selected_k"]


def rk(d: str, s: str) -> str:
    """Resolve the ``kSEL`` / ``KSEL`` placeholders of a kNN condition name or label to the k
    selected on VAL in domain ``d``."""
    for v, k in knn_sel(d).items():
        s = s.replace(f"knn_{v}_kSEL", f"knn_{v}_k{k}")
    return s.replace("KSEL", "k")


def _content_or_lex(R: Results, d: str, cond: str, kind: str) -> pd.Series:
    cond = rk(d, cond)
    if kind == "qn":
        return R.qn.loc[(d, GTE, TC, cond)]
    return {"lex": R.lex, "knn": R.knn, "ctl": R.ctl}[kind].loc[(d, TC, cond)]


def _test(df: pd.DataFrame, d: str, test: str, family: str | None = None, metric="ndcg10"):
    kw = dict(domain=d, text=TC, metric=metric, test=test)
    if family:
        kw["family"] = family
    return _stat_row(df, **kw)


LEX_ROWS = (  # (condition, label, source)
    ("popularity", "Popularity list", "qn"),
    ("raw", "Raw", "qn"),
    ("center_train_users", "Centred query (users)", "qn"),
    ("zca_train_users_a1.0", "ZCA of the query (users)", "qn"),
    ("bm25_title", "BM25, title", "lex"),
    ("bm25_titlecat", "BM25, title + categories", "lex"),
    ("idf_overlap_title", "IDF overlap, title", "lex"),
    ("idf_overlap_titlecat", "IDF overlap, title + categories", "lex"),
)


def table_lexical(R: Results) -> str:
    """Embedding-free lexical matchers against the content conditions (GTE, exact relevance)."""
    body = (
        "Method & Users with a hit & nDCG@10 & R@100 & "
        "$\\Delta$ vs centred query [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ & "
        "$\\Delta$ vs query ZCA [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
    )
    for di, d in enumerate(DOMAINS):
        if di:
            body += "\\midrule\n"
        n = int(qn_row(R, d, "raw")["n_users"])
        body += f"\\multicolumn{{10}}{{l}}{{\\textbf{{{DNAME[d]}}} ($n={thousands(n)}$ test users)}} \\\\\n"
        for cond, label, kind in LEX_ROWS:
            r = _content_or_lex(R, d, cond, kind)
            cells = [f"{int(r['n_hit10']):,}", fnum(r["ndcg10"], 4), fnum(r["r100"], 4)]
            for ref in ("center_train_users", "zca_train_users_a1.0"):
                t = (
                    _test(R.lex_stats, d, f"{cond}_vs_{ref}", "lexical_vs_content")
                    if kind == "lex"
                    else None
                )
                cells += (
                    ["--", "--", "--"]
                    if t is None
                    else [dci(t["mean_diff"], t["ci95_low"], t["ci95_high"]),
                          f"{int(t['n_nonzero_diff']):,}", pval(t["p_holm_signflip"])]
                )  # fmt: skip
            body += f"\\quad {label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "lrrrrrrrrr")


def table_lexical_desc(R: Results) -> str:
    """Lexical matchers over the long music text (desc) against the desc-embedding conditions."""
    refs = (("center_train_users", "centred query"), ("zca_train_users_a1.0", "query ZCA"),
            ("raw", "raw"), ("popularity", "popularity"))  # fmt: skip
    body = (
        "Method & nDCG@10 & "
        + " & ".join(
            f"$\\Delta$ vs {lab} [95\\% CI] & $n_\\Delta$ & $p_{{\\mathrm{{Holm}}}}$"
            for _, lab in refs
        )
        + " \\\\\n\\midrule\n"
    )
    for cond, label in (("popularity", "Popularity list"), ("raw", "Raw"),
                        ("center_train_users", "Centred query (users)"),
                        ("zca_train_users_a1.0", "ZCA of the query (users)")):  # fmt: skip
        body += (
            f"{label} & {fnum(R.qn.loc[('music', GTE, 'desc', cond)]['ndcg10'], 4)}"
            + " & --" * 12
            + " \\\\\n"
        )
    for cond, label in (
        ("bm25_desc", "BM25, long text"),
        ("idf_overlap_desc", "IDF overlap, long text"),
    ):
        cells = [fnum(R.lex.loc[("music", "desc", cond)]["ndcg10"], 4)]
        for ref, _ in refs:
            t = _stat_row(R.lex_stats, domain="music", text="desc", metric="ndcg10",
                          family="lexical_vs_content", test=f"{cond}_vs_{ref}")  # fmt: skip
            cells += [dci(t["mean_diff"], t["ci95_low"], t["ci95_high"]),
                      f"{int(t['n_nonzero_diff']):,}", pval(t["p_holm_signflip"])]  # fmt: skip
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "lr" + "rrr" * 4)


CTL_ROWS = (
    ("raw", "Raw"),
    ("center_train_users", "Centred query (users)"),
    ("zca_train_users_a1.0", "ZCA of the query (users)"),
    ("zca_catalog_a1.0", r"ZCA of the catalog ($\alpha{=}1$)"),
    ("bm25_title", "BM25, title"),
    ("bm25_titlecat", "BM25, title + categories"),
)


def table_controls(R: Results) -> str:
    """Shuffled-profile control: each test user ranked with another test user's profile."""
    body = (
        " & \\multicolumn{2}{c}{Exact nDCG@10} & \\multicolumn{2}{c}{Users with a hit} & "
        "\\multicolumn{3}{c}{Shuffled $-$ real} & \\multicolumn{2}{c}{$d_2$ nDCG@10} \\\\\n"
        "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-8}\\cmidrule(lr){9-10}\n"
        "Condition & real & shuffled & real & shuffled & $\\Delta\\times10^{3}$ [95\\% CI] & $n_\\Delta$ & "
        "$p_{\\mathrm{Holm}}$ & real & shuffled \\\\\n\\midrule\n"
    )
    for di, d in enumerate(DOMAINS):
        if di:
            body += "\\midrule\n"
        body += f"\\multicolumn{{10}}{{l}}{{\\textbf{{{DNAME[d]}}}}} \\\\\n"
        for cond, label in CTL_ROWS:
            a, b = R.ctl.loc[(d, TC, cond)], R.ctl.loc[(d, TC, f"shuffled_{cond}")]
            t = _test(R.ctl_stats, d, f"shuffled_{cond}_vs_{cond}")
            body += (
                f"\\quad {label} & {fnum(a['ndcg10'], 4)} & {fnum(b['ndcg10'], 4)} & "
                f"{int(a['n_hit10']):,} & {int(b['n_hit10']):,} & "
                f"{dci(t['mean_diff'], t['ci95_low'], t['ci95_high'])} & {int(t['n_nonzero_diff']):,} & "
                f"{pval(t['p_holm_signflip'])} & {fnum(a['d2_ndcg10'], 3)} & {fnum(b['d2_ndcg10'], 3)} \\\\\n"
            )
    return wrap(body, "lrrrrrrrrr")


CW_ROWS = (
    ("popularity", "Popularity list", "qn"),
    ("knn_center_train_users_kSEL", "Profile kNN (centred, $k$ chosen on VAL)", "knn"),
    ("raw", "Raw", "qn"),
    ("center_train_users", "Centred query (users)", "qn"),
    ("zca_train_users_a1.0", "ZCA of the query (users)", "qn"),
    ("zca_catalog_a1.0", r"ZCA of the catalog ($\alpha{=}1$)", "qn"),
    ("bm25_titlecat", "BM25, title + categories", "lex"),
    ("idf_overlap_title", "IDF overlap, title", "lex"),
)


def table_coldwarm(R: Results) -> str:
    """Relevant items split by whether any training user consumed them: cold (nobody) and warm."""
    head = (
        " & "
        + " & ".join(
            f"\\multicolumn{{3}}{{c}}{{{DNAME[d]} ($n_{{\\mathrm{{cold}}}}={thousands(int(qn_row(R, d, 'raw')['cold_n_users']))}$ of "
            f"{thousands(int(qn_row(R, d, 'raw')['n_users']))})}}"
            for d in DOMAINS
        )
        + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3))
        + "\nMethod"
        + " & cold nDCG@10 & cold hits & warm nDCG@10" * 3
        + " \\\\\n"
        + " & ($\\times10^{3}$) & (users) & ($\\times10^{3}$)" * 3
        + " \\\\\n\\midrule\n"
    )
    body = head
    for cond, label, kind in CW_ROWS:
        cells = []
        for d in DOMAINS:
            r = _content_or_lex(R, d, cond, kind)
            cells += [
                fnum(1e3 * r["cold_ndcg10"], 2),
                f"{int(r['cold_n_hit10']):,}",
                fnum(1e3 * r["warm_ndcg10"], 2),
            ]
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "rrr" * 3)


def hits_from_csv(name: str, cond: str, gt: str = "exact") -> int:
    df = pd.read_csv(GRAN / name, usecols=["split", "gt", "condition", "nDCG@10"])
    df = df[(df["split"] == "test") & (df["gt"] == gt) & (df["condition"] == cond)]
    return int((df["nDCG@10"] > 0).sum())


KNN_ROWS = (  # (condition, label, source, lambda key)
    ("popularity", "Popularity list", "knn"),
    ("knn_raw_kSEL", "Profile kNN, raw ($k$ chosen on VAL)", "knn"),
    ("knn_center_train_users_kSEL", "Profile kNN, centred ($k$ chosen on VAL)", "knn"),
    ("zca_train_users_a1.0", "ZCA of the query (users)", "knn"),
    ("hybrid_zca_profile", "ZCA + popularity", "hyb"),
    ("hybrid_knn_raw+zca_train_users_a1.0", "kNN raw + ZCA", "knn"),
    ("hybrid_knn_center_train_users+raw", "kNN centred + raw", "knn"),
    ("hybrid_knn_center_train_users+zca_train_users_a1.0", "kNN centred + ZCA", "knn"),
)


def table_knn(R: Results) -> str:
    """Collaborative profile-kNN over the training users, with and without content (GTE)."""
    body = (
        "Method & Users with a hit & nDCG@10 & R@100 & "
        "$\\Delta$ vs popularity [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ & "
        "$\\Delta$ vs its kNN [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ \\\\\n\\midrule\n"
    )
    for di, d in enumerate(DOMAINS):
        if di:
            body += "\\midrule\n"
        n = int(qn_row(R, d, "raw")["n_users"])
        body += f"\\multicolumn{{10}}{{l}}{{\\textbf{{{DNAME[d]}}} ($n={thousands(n)}$ test users)}} \\\\\n"
        vals = {}
        for cond, _label, kind in KNN_ROWS:
            cond = rk(d, cond)
            if kind == "hyb":
                m = hybrid_json(d)["test"]["exact"][cond]
                hits = hits_from_csv(f"hybrid_{d}_gte.csv.gz", cond)
                vals[cond] = (hits, m["nDCG@10"], m["R@100"])
            else:
                r = R.knn.loc[(d, TC, cond)]
                vals[cond] = (int(r["n_hit10"]), r["ndcg10"], r["r100"])
        bestn = max(v[1] for v in vals.values())
        for cond, label, kind in KNN_ROWS:
            cond = rk(d, cond)
            hits, nd, r100 = vals[cond]
            cells = [f"{hits:,}", bold(fnum(nd, 4), nd >= bestn), fnum(r100, 4)]
            if kind == "hyb":
                t = _stat_row(
                    R.hyb_stats, domain=d, metric="ndcg10", condition=cond, reference="popularity"
                )
            elif cond == "zca_train_users_a1.0":  # content only: the grid test against popularity
                t = _stat_row(R.vs_pop, domain=d, encoder=GTE, text=TC, gt="exact", metric="ndcg10",
                              condition="zca_profile")  # fmt: skip
            else:
                t = _test(R.knn_stats, d, f"{cond}_vs_popularity")
            cells += (
                ["--"] * 3 if t is None
                else [dci(t["mean_diff"], t["ci95_low"], t["ci95_high"]),
                      f"{int(t['n_nonzero_diff']):,}", pval(t["p_holm_signflip"])]
            )  # fmt: skip
            t2 = None
            if cond.startswith("hybrid_knn_"):
                own = rk(d, "knn_" + cond.split("hybrid_knn_")[1].split("+")[0] + "_kSEL")
                t2 = _test(R.knn_stats, d, f"{cond}_vs_{own}")
            if (
                t2 is not None and t2["n_nonzero_diff"] == 0
            ):  # the chosen weight made it the kNN itself
                cells += ["same ranking as its kNN", "0", "--"]
            else:
                cells += (
                    ["--"] * 3 if t2 is None
                    else [dci(t2["mean_diff"], t2["ci95_low"], t2["ci95_high"]),
                          f"{int(t2['n_nonzero_diff']):,}", pval(t2["p_holm_signflip"])]
                )  # fmt: skip
            body += f"\\quad {label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "lrrrrrrrrr")


# ───────────────────────────────── Text format (E6) ────────────────────────────────

TEXT_ROWS = (
    ("title", "Title only"),
    ("titlecat", "Title + categories"),
    ("desc", "Title + categories + description"),
)


def table_text_format(R: Results) -> str:
    """GTE, music: three item-text formats. Exact nDCG@10 (x1000) of raw, the query normalised
    (users, books), the catalog whitened, the contrast (a1) and the depth-2 nDCG@10."""
    body = (
        "Item text & \\multicolumn{5}{c}{Exact nDCG@10 ($\\times10^{3}$)} & "
        "\\multicolumn{3}{c}{(a1) query ZCA vs catalog ZCA} & \\multicolumn{3}{c}{$d_2$ nDCG@10} \\\\\n"
        "\\cmidrule(lr){2-6}\\cmidrule(lr){7-9}\\cmidrule(lr){10-12}\n"
        " & raw & centre & ZCA & ZCA & ZCA & $\\Delta\\times10^{3}$ [95\\% CI] & $n_\\Delta$ & $p_{\\mathrm{Holm}}$ & "
        "raw & query ZCA & catalog ZCA \\\\\n"
        " & & (users) & (users) & (books) & (catalog) & & & & & (users) & \\\\\n\\midrule\n"
    )
    for text, label in TEXT_ROWS:
        cells = [fnum(1e3 * qn_row(R, "music", "raw", GTE, text)["ndcg10"], 2)]
        for c, _item in (("center_train_users", False), ("zca_train_users_a1.0", False),
                        ("zca_books_catalog_a1.0", False), ("zca_catalog_a1.0", False)):  # fmt: skip
            v = qn_row(R, "music", c, GTE, text)["ndcg10"]
            cells.append(fnum(1e3 * v, 2) + qn_mark(R, "music", c, False, GTE, text))
        r = qn_contrast(R, "music", GTE, text, "a_zca_train_users_vs_zca_catalog")
        cells += [dci(r["mean_diff"], r["ci95_low"], r["ci95_high"]), f"{int(r['n_nonzero_diff']):,}",
                  pval(r["p_holm_signflip"])]  # fmt: skip
        for c in ("raw", "zca_train_users_a1.0", "zca_catalog_a1.0"):
            cells.append(fnum(qn_row(R, "music", c, GTE, text)["d2_ndcg10"], 3))
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "r" * 11)


# ─────────────────────────── Dose in the exponent, both sides ──────────────────────

ALPHA_ARMS = (  # (alpha label, query arm (users), catalog arm)
    ("0", "center_train_users", "center_catalog"),
    ("0.25", "zca_train_users_a0.25", "zca_catalog_a0.25"),
    ("0.5", "zca_train_users_a0.5", "zca_catalog_a0.5"),
    ("0.75", "zca_train_users_a0.75", "zca_catalog_a0.75"),
    ("1", "zca_train_users_a1.0", "zca_catalog_a1.0"),
)


def alpha_gain(R: Results, d: str, cond: str, level: str) -> float:
    """Relative nDCG@10 gain over raw at ``level`` (exact or d2) of a query-normalisation arm."""
    key = "ndcg10" if level == "exact" else "d2_ndcg10"
    raw = qn_row(R, d, "raw")[key]
    return float(qn_row(R, d, cond)[key] / raw - 1)


def table_alpha(R: Results) -> str:
    """Relative gain over raw by exponent alpha (alpha=0 is centring only): the query
    (statistics of the users) and the catalog (its own statistics). Markers only where a test
    against raw exists (alpha 0 and 1 of the query side; every catalog arm)."""
    body = (
        "Domain & Level & "
        + " & ".join(f"\\multicolumn{{2}}{{c}}{{$\\alpha={a}$}}" for a, _, _ in ALPHA_ARMS)
        + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{3 + 2 * i}-{4 + 2 * i}}}" for i in range(5))
        + "\n & & "
        + " & ".join("query & catalog" for _ in ALPHA_ARMS)
        + " \\\\\n\\midrule\n"
    )
    for di, d in enumerate(DOMAINS):
        if di:
            body += "\\midrule\n"
        for gi, g in enumerate(("exact", "d2")):
            cells = []
            for _, qa, ca in ALPHA_ARMS:
                qm = qn_mark(R, d, qa, False) if g == "exact" else ""
                cm = _delta_cell(R, GTE, g, ca, d)[0] if g in levels_of(d) else "--"
                qv = signed(100 * alpha_gain(R, d, qa, g), 0) + (STAR if qm else "")
                cells += [qv, cm]
            body += f"{DNAME[d] if gi == 0 else ''} & {g} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "ll" + "rr" * 5)


# ───────────────── Matrix tally, validation replication, splits, lexical overlap ─────────────────


def matrix_tally_rows(R: Results) -> str:
    """Rows of Table 6 for the six cells of the side x source matrix (ZCA, alpha 1) against raw:
    better / not significant / worse (Holm sign-flip p < 0.05, family ``matrix``) over the
    verified runs of the title+categories text (eight music encoders, GTE in movies and games)."""
    t = R.qn_tally
    t = t[(t["text"] == TC) & (t["metric"] == "ndcg10") & (t["family"] == "matrix")]
    t = t[t["encoders"] == "verified"]
    body = ""
    for (side, src), _ in MATRIX_KEY.items():
        sub = t[t["test"] == f"matrix:{side}_{src}_vs_raw"]
        b, ns, w, n = (int(sub[c].sum()) for c in ("better", "ns", "worse", "n_tests"))
        lab = f"{'Query' if side == 'query' else 'Items'} ZCA, {SRC_NAME[src]}"
        body += f"{lab} & verified & {n} & {n} & {b} / {ns} / {w} & -- & -- & -- \\\\\n"
    return body


SRC_NAME = {"U": "users", "B": "books", "C": "target"}


def _val_cell(d: pd.DataFrame, domain: str, test: str, family: str) -> pd.Series:
    r = _stat_row(d, domain=domain, encoder=GTE, text=TC, metric="ndcg10", family=family, test=test)
    assert r is not None, (domain, test)
    return r


VAL_ROWS = (  # (family, test, label)
    ("contrast", "a_zca_train_users_vs_zca_catalog", r"(a1) ZCA query (users) $-$ ZCA items"),
    ("contrast", "a_center_train_users_vs_zca_catalog", r"(a2) centring query (users) $-$ ZCA items"),
    ("contrast", "a_zca_books_catalog_vs_zca_catalog", r"(a3) ZCA query (books) $-$ ZCA items"),
    ("contrast", "b_zca_vs_center_train_users", r"(b) ZCA $-$ centring, query (users)"),
    ("contrast", "c_both_sides_vs_query_only", r"(c) both sides $-$ query only"),
    ("post_hoc", "post_hoc:zca_target_catalog_vs_raw", r"(d1)$^{\ddagger}$ ZCA query (target) $-$ raw"),
    ("post_hoc", "post_hoc:zca_train_users_vs_zca_target_catalog", r"(d2)$^{\ddagger}$ ZCA query (users) $-$ ZCA query (target)"),
    ("matrix", "matrix:query_U_vs_raw", r"ZCA query (users) $-$ raw"),
    ("matrix", "matrix:items_U_vs_raw", r"(m1)$^{\S}$ ZCA items (users) $-$ raw"),
    ("matrix", "matrix:side_U", r"(m2)$^{\S}$ query $-$ items, users' statistics"),
    ("matrix", "matrix:side_B", r"(m3)$^{\S}$ query $-$ items, books' statistics"),
    ("matrix", "matrix:side_C", r"(m4)$^{\S}$ query $-$ items, target statistics"),
)  # fmt: skip


def replicates(test_row: pd.Series, val_row: pd.Series) -> bool:
    """The contrast replicates when it is significant (Holm sign-flip p < 0.05) on TEST AND on VAL
    with the same sign; a null on both does not count as a replication."""
    t = mark(test_row["p_holm_signflip"], test_row["mean_diff"])
    v = mark(val_row["p_holm_signflip"], val_row["mean_diff"])
    return t != "" and t == v


def ci_holm_disagree(row: pd.Series) -> bool:
    """The unadjusted bootstrap 95% CI excludes 0 while the Holm sign-flip p is >= 0.05 (or the
    reverse): the CI is a percentile bootstrap, the p a sign-flip adjusted within the family."""
    excl = not (row["ci95_low"] <= 0 <= row["ci95_high"])
    return excl != (row["p_holm_signflip"] < ALPHA)


CIFLAG = r"$^{\circ}$"  # marks a CI / Holm disagreement (see ci_holm_disagree)


def table_val_replication(R: Results) -> str:
    """Contrasts of the query-normalisation family on the TEST and on the VAL users (GTE)."""
    head = (
        "Contrast & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3))
        + "\n & "
        + " & ".join(
            "test $\\Delta$ & val $\\Delta$ [95\\% CI] & val $p_{\\mathrm{Holm}}$" for _ in DOMAINS
        )
        + " \\\\\n\\midrule\n"
    )
    body = head
    for fam, test, label in VAL_ROWS:
        cells = []
        for d in DOMAINS:
            t = _val_cell(R.qn_stats, d, test, fam)
            v = _val_cell(R.qn_val, d, test, fam)
            cells += [
                signed(1e3 * t["mean_diff"], 2) + mark(t["p_holm_signflip"], t["mean_diff"]),
                dci(v["mean_diff"], v["ci95_low"], v["ci95_high"])
                + mark(v["p_holm_signflip"], v["mean_diff"])
                + (CIFLAG if ci_holm_disagree(v) else ""),
                pval(v["p_holm_signflip"]),
            ]
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "rrr" * 3)


MS_MAIN = (  # contrast ids of the compact multi-split table, in order
    "qU_vs_raw", "qCtr_vs_raw", "a1", "a4", "b", "c", "itemsU_vs_raw", "sideU", "sideB", "sideC",
    "zca_vs_pop",
    "hybZca_vs_pop", "knnHyb_vs_knn", "knnHyb_vs_pop", "lexIdf_vs_ctr", "lexBm25_vs_zca",
)  # fmt: skip


def _ms_cell(R: Results, cid: str, d: str) -> tuple[str, str, str]:
    r = _stat_row(R.ms, contrast=cid, domain=d)
    assert r is not None, (cid, d)
    mean, sd = 1e3 * r["mean_diff_mean"], 1e3 * r["mean_diff_sd"]
    return (
        f"{signed(mean, 2)} $\\pm$ {fnum(sd, 2)}",
        f"{int(r['n_better'])} / {int(r['n_ns'])} / {int(r['n_worse'])}",
        f"{signed(1e3 * r['mean_diff_min'], 2)} / {signed(1e3 * r['mean_diff_max'], 2)}",
    )


def table_multisplit(R: Results) -> str:
    """Key contrasts on the five splits: mean and standard deviation (x10^3) of the paired mean
    difference over the splits and the number of splits where it is significantly better / not
    significant / significantly worse (Holm sign-flip p < 0.05 within the split's own family), and
    the minimum / maximum of the per-split difference."""
    labels = dict(R.ms.drop_duplicates("contrast").set_index("contrast")["label"])
    head = (
        "Contrast & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(3))
        + "\n & "
        + " & ".join("mean $\\pm$ sd ($\\times10^{3}$) & b / ns / w & min / max" for _ in DOMAINS)
        + " \\\\\n\\midrule\n"
    )
    body = head
    for cid in MS_MAIN:
        cells = [c for d in DOMAINS for c in _ms_cell(R, cid, d)]
        body += f"{tex(labels[cid])} & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n" + "Centring share of the query-ZCA gain (\\%)"
    for d in DOMAINS:
        r = R.ms_share[R.ms_share["domain"] == d].iloc[0]
        m, sd, lo, hi = (
            f"{100 * r[k]:.0f}" for k in ("share_mean", "share_sd", "share_min", "share_max")
        )
        body += f" & {m} $\\pm$ {sd} & & {lo} / {hi}"
    body += " \\\\\n"
    return wrap(body, "l" + "rrr" * 3)


def table_multisplit_all(R: Results) -> str:
    """Per-split paired mean differences (x10^3) of every contrast, with the Holm significance
    marker of the split, in the order of the seeds."""
    seeds = sorted(R.ms_ps["seed"].unique())
    body = (
        "Contrast & "
        + " & ".join(f"\\multicolumn{{{len(seeds)}}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n"
        + "".join(
            f"\\cmidrule(lr){{{2 + len(seeds) * i}-{1 + len(seeds) * (i + 1)}}}" for i in range(3)
        )
        + "\n & "
        + " & ".join(str(s) for _ in DOMAINS for s in seeds)
        + " \\\\\n\\midrule\n"
    )
    labels = dict(R.ms.drop_duplicates("contrast").set_index("contrast")["label"])
    order = [c for c in R.ms["contrast"].drop_duplicates()]
    for cid in order:
        cells = []
        for d in DOMAINS:
            for sd in seeds:
                r = _stat_row(R.ms_ps, contrast=cid, domain=d, seed=sd)
                cells.append(
                    "--" if r is None else
                    signed(1e3 * r["mean_diff"], 2) + mark(r["p_holm_signflip"], r["mean_diff"])
                )  # fmt: skip
        body += f"{tex(labels[cid])} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "r" * (len(seeds) * 3))


LO_SCORERS = (
    ("idf_overlap_title", "IDF overlap"),
    ("bm25_title", "BM25"),
    ("strict_idf_h2_window", "Strict IDF overlap"),
)
LO_ARMS = (("raw", "Raw"), ("center_train_users", "Centred"), ("zca_train_users_a1.0", "ZCA"))


def table_overlap(R: Results, arms=("zca_train_users_a1.0",)) -> str:
    """Users with a top-10 hit under a query arm and under a lexical scorer: users with a hit
    under each, under both, the number expected under independence (overall and inside strata of
    the number of relevant items), the Jaccard index and the share of the arm's hit users that
    the scorer also serves."""
    head = (
        "Domain & Arm & Lexical scorer & arm hits & lexical hits & both & chance & chance (strata) "
        "& Jaccard & share of arm hits also lexical \\\\\n\\midrule\n"
    )
    body = head
    names = dict(LO_ARMS)
    for d in DOMAINS:
        first = True
        for arm in arms:
            for lex, ll in LO_SCORERS:
                r = R.lo[d]["A_hit_overlap"][arm][lex]
                body += (
                    f"{DNAME[d] if first else ''} & {names[arm]} & {ll} & {r['n_a']} & {r['n_b']} & "
                    f"{r['both']} & {r['expected_independent']:.1f} & {r['expected_stratified']:.1f} & "
                    f"{r['jaccard']:.2f} & {100 * r['share_a_in_b']:.0f}\\% \\\\\n"
                )
                first = False
        body += "\\addlinespace\n"
    return wrap(body, "lll" + "r" * 7)


LO_RANK = (
    ("raw", "Raw"), ("center_train_users", "Centred query"), ("zca_train_users_a1.0", "ZCA of the query"),
    ("idf_overlap_title", "IDF overlap"), ("bm25_title", "BM25"),
    ("strict_idf_h2_window", "Strict IDF overlap"), ("popularity", "Popularity"),
)  # fmt: skip


def table_reach(R: Results, frac: str = "0.01") -> str:
    """Share of the hit pairs (relevant item among a user's top 10) whose item shares a rare title
    token with the books of the user's window, with the number of hit pairs; below, the chance
    levels: the share among the relevant items and the share among catalog items."""
    body = (
        "Ranking & "
        + " & ".join(f"\\multicolumn{{1}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n\\midrule\n"
    )
    for cond, label in LO_RANK:
        cells = []
        for d in DOMAINS:
            r = R.lo[d]["B_reachability"][frac]["rankings"][cond]
            cells.append(f"{100 * r['hit_reachable_share']:.0f}\\% ({r['hit_pairs']})")
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n"
    for key, label in (
        ("relevant_reachable_share", "Chance: relevant items"),
        ("catalog_reachable_share", "Chance: catalog items"),
    ):
        cells = [f"{100 * R.lo[d]['B_reachability'][frac][key]:.1f}\\%" for d in DOMAINS]
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "lrrr")


def _lo_row(R: Results, d: str, gt: str, cond: str) -> pd.Series:
    r = _stat_row(R.lo_sum, domain=d, gt=gt, condition=cond)
    assert r is not None, (d, gt, cond)
    return r


def _lo_test(R: Results, d: str, gt: str, test: str) -> pd.Series:
    r = _stat_row(R.lo_stats, domain=d, gt=gt, metric="ndcg10", test=test)
    assert r is not None, (d, gt, test)
    return r


LO_TESTS = (
    ("zca_vs_raw", r"ZCA query $-$ raw"),
    ("center_vs_raw", r"Centred query $-$ raw"),
    ("zca_vs_center", r"ZCA query $-$ centred query"),
    ("zca_vs_idf_overlap_title", r"ZCA query $-$ IDF overlap"),
    ("zca_vs_bm25_title", r"ZCA query $-$ BM25"),
)


def table_residual(R: Results) -> str:
    """nDCG@10 (x1000) against the residual relevant set (no shared rare title token with the
    user's books) and against the reachable one, with the number of users with a hit, and paired
    tests on the residual set."""
    head = (
        "Ranking & "
        + " & ".join(f"\\multicolumn{{4}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{2 + 4 * i}-{5 + 4 * i}}}" for i in range(3))
        + "\n & "
        + " & ".join("resid. nDCG & hit users & reach. nDCG & hit users" for _ in DOMAINS)
        + " \\\\\n\\midrule\n"
    )
    body = head
    for cond, label in (*LO_RANK, ("random", "Random")):
        cells = []
        for d in DOMAINS:
            a, b = _lo_row(R, d, "resid", cond), _lo_row(R, d, "reach", cond)
            cells += [
                fnum(1e3 * a["nDCG@10"], 2),
                str(int(a["n_hit10"])),
                fnum(1e3 * b["nDCG@10"], 2),
                str(int(b["n_hit10"])),
            ]
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    body += "\\midrule\n"
    body += (
        "Paired tests on the residual set & "
        + " & ".join(
            "\\multicolumn{3}{c}{$\\Delta\\times10^{3}$ [95\\% CI]} & $n_\\Delta$" for _ in DOMAINS
        )
        + " \\\\\n"
    )
    for test, label in LO_TESTS:
        cells = []
        for d in DOMAINS:
            t = _lo_test(R, d, "resid", test)
            d_ci = dci(t["mean_diff"], t["ci95_low"], t["ci95_high"])
            cells += [
                f"\\multicolumn{{3}}{{c}}{{{d_ci}{mark(t['p_holm_signflip'], t['mean_diff'])}"
                f"{CIFLAG if ci_holm_disagree(t) else ''}}}",
                f"{int(t['n_nonzero_diff'])}",
            ]
        body += f"{label} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "l" + "rrrr" * 3)


MASK_FRACS = (("resid@0.001", "0.1\\%"), ("resid", "1\\%"), ("resid@0.05", "5\\%"))
MASK_TESTS = (
    ("masked_zca_vs_raw", r"ZCA query $-$ raw"),
    ("masked_center_vs_raw", r"Centred query $-$ raw"),
    ("masked_zca_vs_center", r"ZCA query $-$ centred query"),
)


def table_residual_masked(R: Results) -> str:
    """Masked residual: the user's reachable items are removed from the ranking and the arms are
    re-ranked over the rest (nDCG@10 against the residual relevant set), for rarity thresholds of
    0.1%, 1% and 5% of the catalog titles. Paired difference x10^3 [95% CI, unadjusted], the
    minimum detectable difference at 80% power (same units) and the number of users whose two
    values differ; a circle marks a CI that disagrees with the Holm test."""
    head = (
        "Threshold & Contrast & "
        + " & ".join(f"\\multicolumn{{3}}{{c}}{{{DNAME[d]}}}" for d in DOMAINS)
        + " \\\\\n"
        + "".join(f"\\cmidrule(lr){{{3 + 3 * i}-{5 + 3 * i}}}" for i in range(3))
        + "\n & & "
        + " & ".join("$\\Delta\\times10^{3}$ [95\\% CI] & MDE & $n_\\Delta$" for _ in DOMAINS)
        + " \\\\\n\\midrule\n"
    )
    body = head
    for gt, tl in MASK_FRACS:
        for i, (test, label) in enumerate(MASK_TESTS):
            cells = []
            for d in DOMAINS:
                t = _lo_test(R, d, gt, test)
                cells += [
                    dci(t["mean_diff"], t["ci95_low"], t["ci95_high"])
                    + mark(t["p_holm_signflip"], t["mean_diff"])
                    + (CIFLAG if ci_holm_disagree(t) else ""),
                    fnum(1e3 * t["mde80"], 2),
                    str(int(t["n_nonzero_diff"])),
                ]
            body += f"{tl if i == 0 else ''} & {label} & " + " & ".join(cells) + " \\\\\n"
        body += "\\addlinespace\n"
    return wrap(body, "ll" + "rrr" * 3)


def table_overlap_null(R: Results) -> str:
    """Share of the hit users of each query arm (raw, centred, ZCA) that also hit under a lexical
    scorer, next to the share expected when the two hit sets are independent inside strata of the
    number of relevant items: ZCA's overlap is compared with raw's, not only with independence."""
    head = (
        "Domain & Lexical scorer & "
        + " & ".join(f"{n} share" for _, n in LO_ARMS)
        + " & chance (ZCA strata) \\\\\n\\midrule\n"
    )
    body = head
    for d in DOMAINS:
        for j, (lex, ll) in enumerate(LO_SCORERS):
            A = R.lo[d]["A_hit_overlap"]
            cells = [f"{100 * A[a][lex]['share_a_in_b']:.0f}\\%" for a, _ in LO_ARMS]
            z = A["zca_train_users_a1.0"][lex]
            cells.append(f"{100 * z['expected_stratified'] / z['n_a']:.0f}\\%")
            body += f"{DNAME[d] if j == 0 else ''} & {ll} & " + " & ".join(cells) + " \\\\\n"
        body += "\\addlinespace\n"
    return wrap(body, "ll" + "r" * 4)


def _dcell(t: pd.Series) -> str:
    """Paired difference x10^3 [95% CI] with the Holm arrow and, when the CI and the Holm test
    disagree, the circle."""
    return (dci(t["mean_diff"], t["ci95_low"], t["ci95_high"])
            + mark(t["p_holm_signflip"], t["mean_diff"])
            + (CIFLAG if ci_holm_disagree(t) else ""))  # fmt: skip


def table_bridge(R: Results, R2: Results | None = None) -> str:
    """EMCDR-style bridges (book profile, for the movies pairs the movie profile, to the target
    embedding space): closed-form ridge, one-hidden-layer MLP (MSE to the mean embedding of the
    user's train items) and the contrastive MLP (InfoNCE, mean of 3 seeds; plain and with the
    query centred after the bridge), all with their hyper-parameters chosen on VAL, against raw
    and the ZCA query: TEST exact nDCG@10 (x1000) and paired differences x10^3 [95% CI]. A circle
    marks a CI that disagrees with the Holm test. Last block: sanity check, nDCG@10 (x1000) of
    one held-out train target item per train user for the contrastive bridge, raw, query ZCA and
    a random ranking."""
    head = (
        "Pair & ridge & MLP & CL & CL centred & raw & ZCA query & ridge $-$ raw & MLP $-$ raw"
        " & CL $-$ raw & CL $-$ ZCA query & CL $-$ ridge & CL centred $-$ ZCA query \\\\\n\\midrule\n"
    )
    body = head
    R2 = R2 or Results("_source2")
    allp = []
    for res, doms, names in ((R, DOMAINS, DNAME), (R2, DOMAINS2, DNAME2)):
        for d in doms:
            allp.append((d, names[d]))
            cells = [fnum(1e3 * res.br.loc[(d, TC, c)]["ndcg10"], 2) for c in
                     ("bridge_ridge", "bridge_mlp", "bridge_cl", "bridge_cl_centred", "raw",
                      "zca_train_users_a1.0")]  # fmt: skip
            for test in ("bridge_ridge_vs_raw", "bridge_mlp_vs_raw", "bridge_cl_vs_raw",
                         "bridge_cl_vs_zca_train_users_a1.0", "bridge_cl_vs_bridge_ridge",
                         "bridge_cl_centred_vs_zca_train_users_a1.0"):  # fmt: skip
                cells.append(_dcell(_stat_row(res.br_stats, domain=d, metric="ndcg10", test=test)))
            body += f"{names[d]} & " + " & ".join(cells) + " \\\\\n"
        if res is R:
            body += "\\midrule\n"
    body += ("\\midrule\n\\multicolumn{13}{l}{Sanity: nDCG@10 $\\times10^{3}$ of one held-out train "
             "target item per train user, model refitted without it} \\\\\n")  # fmt: skip
    for d, name in allp:
        s_ = json.loads((GRAN / f"bridgecl_{d}_gte.json").read_text())["sanity"]
        body += (f"{name} & \\multicolumn{{12}}{{l}}{{contrastive {fnum(1e3 * s_['bridge_cl'], 2)}; "
                 f"raw {fnum(1e3 * s_['raw'], 2)}; ZCA query {fnum(1e3 * s_['zca_train_users_a1.0'], 2)}; "
                 f"random {fnum(1e3 * s_['random'], 2)} ($n={thousands(s_['n_users'])}$ users)}} \\\\\n")  # fmt: skip
    return wrap(body, "l" + "r" * 12)


def table_source2(R: Results, R2: Results | None = None) -> str:
    """Movies as SOURCE (movies to music, movies to games), GTE, exact nDCG@10 x1000 on the TEST
    users: baselines, then the side x source matrix of ZCA (alpha 1): the query or the items
    transformed with the statistics of the training users U, the source (movie) catalog M or the
    target catalog C, and the matched-source side contrast (query minus items). Arrows: Holm
    sign-flip against raw."""
    R2 = R2 or Results("_source2")
    D = DOMAINS2
    n = {d: int(qn_row(R2, d, "raw")["n_users"]) for d in D}
    body = (
        "Condition & "
        + " & ".join(f"{DNAME2[d]} ($n={thousands(n[d])}$)" for d in D)
        + " \\\\\n\\midrule\n"
    )

    def row(label: str, cells: list[str]) -> str:
        return f"{label} & " + " & ".join(cells) + " \\\\\n"

    ks = {d: json.loads((GRAN / f"knn_{d}_gte.json").read_text())["selected_k"] for d in D}
    for label, getter in (
        ("Popularity list", lambda d: qn_row(R2, d, "popularity")),
        ("Raw (no correction)", lambda d: qn_row(R2, d, "raw")),
        ("Profile kNN, raw (k on VAL)", lambda d: R2.knn.loc[(d, TC, f"knn_raw_k{ks[d]['raw']}")]),
        ("Profile kNN, centred (k on VAL)", lambda d: R2.knn.loc[(d, TC, f"knn_center_train_users_k{ks[d]['center_train_users']}")]),
        ("BM25, source titles vs target titles", lambda d: R2.lex.loc[(d, TC, "bm25_title")]),
        ("IDF overlap, source titles vs target titles", lambda d: R2.lex.loc[(d, TC, "idf_overlap_title")]),
    ):
        body += row(label, [fnum(1e3 * getter(d)["ndcg10"], 2) for d in D])
    body += "\\midrule\n"
    best = {
        d: max(qn_row(R2, d, MATRIX_KEY[(side, src)])["ndcg10"] for side in ("query", "items")
               for src in SOURCES3)
        for d in D
    }  # fmt: skip
    for side, label in (("query", "Query"), ("items", "Items")):
        for src, sl in (("U", "train users U"), ("B", "movie catalog M"), ("C", "target catalog C")):
            cells = []
            for d in D:
                v = qn_row(R2, d, MATRIX_KEY[(side, src)])["ndcg10"]
                cells.append(bold(fnum(1e3 * v, 2), v >= best[d]) + matrix_mark(R2, d, side, src))
            body += row(f"{label}, statistics of {sl}", cells)
    both = "zca_both_separate"
    body += row("Both sides (query: users U, items: target catalog C)", [
        fnum(1e3 * qn_row(R2, d, both)["ndcg10"], 2) + qn_mark(R2, d, both, False) for d in D])  # fmt: skip
    body += "\\midrule\n"
    for src, sl in (("U", "train users U"), ("B", "movie catalog M"), ("C", "target catalog C")):
        cells = []
        for d in D:
            r = matrix_test(R2, d, f"matrix:side_{src}")
            cells.append(signed(1e3 * r["mean_diff"], 2) + mark(r["p_holm_signflip"], r["mean_diff"]))
        body += row(f"Query $-$ items, {sl}", cells)
    body += "\\midrule\n"
    z = "zca_train_users_a1.0"
    for label, df, test_of in (
        ("Popularity $-$ query ZCA (U)", R2.knn_stats, lambda d: f"popularity_vs_{z}"),
        ("kNN raw $-$ query ZCA (U)", R2.knn_stats, lambda d: f"knn_raw_k{ks[d]['raw']}_vs_{z}"),
        ("kNN centred $-$ query ZCA (U)", R2.knn_stats,
         lambda d: f"knn_center_train_users_k{ks[d]['center_train_users']}_vs_{z}"),
        ("BM25 $-$ query ZCA (U)", R2.lex_stats, lambda d: f"bm25_title_vs_{z}"),
        ("IDF overlap $-$ query ZCA (U)", R2.lex_stats, lambda d: f"idf_overlap_title_vs_{z}"),
    ):  # fmt: skip
        body += row(label, [_dcell(_test(df, d, test_of(d))) for d in D])
    body += row("Both sides $-$ query only (U)", [
        _dcell(qn_contrast(R2, d, GTE, TC, "c_both_sides_vs_query_only")) for d in D])  # fmt: skip
    return wrap(body, "lrr")


def _flip(t: pd.Series) -> pd.Series:
    """The same paired test with the roles exchanged (reference minus condition)."""
    t = t.copy()
    t["mean_diff"], t["ci95_low"], t["ci95_high"] = -t["mean_diff"], -t["ci95_high"], -t["ci95_low"]
    return t


def table_popularity_compact(R: Results) -> str:
    """Popularity list against the content queries and the profile kNN in the three books pairs
    (GTE, TEST exact nDCG@10 x1000); paired differences (content minus popularity) x10^3
    [95% CI], arrows for the Holm sign-flip test, circle where the CI and Holm disagree."""
    body = ("Pair & Popularity & Raw & ZCA query & kNN raw & Raw $-$ pop. & ZCA query $-$ pop."
            " & kNN raw $-$ pop. \\\\\n\\midrule\n")  # fmt: skip
    z = "zca_train_users_a1.0"
    for d in DOMAINS:
        k = knn_sel(d)["raw"]
        cells = [fnum(1e3 * qn_row(R, d, c)["ndcg10"], 2) for c in ("popularity", "raw", z)]
        cells.append(fnum(1e3 * R.knn.loc[(d, TC, f"knn_raw_k{k}")]["ndcg10"], 2))
        cells += [_dcell(_flip(_test(R.knn_stats, d, f"popularity_vs_{c}"))) for c in ("raw", z)]
        cells.append(_dcell(_test(R.knn_stats, d, f"knn_raw_k{k}_vs_popularity",
                                  "knn_vs_references")))  # fmt: skip
        body += f"{DNAME[d]} & " + " & ".join(cells) + " \\\\\n"
    return wrap(body, "lrrrrrrr")


TABLES = {
    "tab_domains": lambda R: table_domains(),
    "tab_main": lambda R: table_conditions(R, DOMAINS, ROWS_MAIN, full=False, inline=True),
    "tab_qnorm": table_qnorm,
    "tab_matrix": table_matrix,
    "tab_qnorm_tests": table_qnorm_tests,
    "tab_encoders": table_encoders,
    "tab_tally": table_tally,
    "tab_lexical": table_lexical,
    "tab_lexical_desc": table_lexical_desc,
    "tab_controls": table_controls,
    "tab_coldwarm": table_coldwarm,
    "tab_knn": table_knn,
    "tab_text_format": table_text_format,
    "tab_mechanism": lambda R: table_mechanism(),
    "tab_signal": lambda R: table_signal(),
    "tab_encoders_gap": table_encoders_gap,
    "tab_full": table_full,
    "tab_encoders_all": table_encoders_all,
    "tab_mechanism_full": lambda R: table_mechanism_full(),
    "tab_baselines": table_baselines,
    "tab_alpha": table_alpha,
    "tab_hybrid": lambda R: table_hybrid(),
    "tab_hybrid_val": lambda R: table_hybrid_val(),
    "tab_val_replication": table_val_replication,
    "tab_multisplit": table_multisplit,
    "tab_multisplit_all": table_multisplit_all,
    "tab_overlap": table_overlap,
    "tab_overlap_other": lambda R: table_overlap(R, ("raw", "center_train_users")),
    "tab_reach": table_reach,
    "tab_residual": table_residual,
    "tab_residual_masked": table_residual_masked,
    "tab_overlap_null": table_overlap_null,
    "tab_bridge": lambda R: table_bridge(R),
    "tab_source2": table_source2,
    "tab_popularity_compact": table_popularity_compact,
}


# Tables whose natural width at \\footnotesize exceeds \\textwidth: the plain tabular is scaled
# down (never up) to exactly \\textwidth. A tabular* cannot be shrunk, its width is fixed.
WIDE = {
    "tab_qnorm_tests", "tab_encoders", "tab_text_format", "tab_lexical", "tab_coldwarm", "tab_knn",
    "tab_multisplit", "tab_lexical_desc", "tab_val_replication", "tab_residual",
    "tab_residual_masked", "tab_hybrid", "tab_signal", "tab_bridge", "tab_source2",
    "tab_popularity_compact",
}  # fmt: skip


def shrink_to_fit(text: str) -> str:
    text = text.replace(
        "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}", "\\begin{tabular}{"
    )
    text = text.replace("\\end{tabular*}", "\\end{tabular}")
    return text.replace(
        "\\begin{tabular}", "\\adjustbox{max width=\\textwidth}{\\begin{tabular}", 1
    ).replace("\\end{tabular}", "\\end{tabular}}", 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="directory that receives tab_*.tex")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = Results()
    for name, fn in TABLES.items():
        text = fn(R)
        if name in WIDE:
            text = shrink_to_fit(text)
        assert not re.search(r"\bnan\b", text), name
        (args.out / f"{name}.tex").write_text(text)
        print(f"wrote {args.out / (name + '.tex')}")


if __name__ == "__main__":
    main()
