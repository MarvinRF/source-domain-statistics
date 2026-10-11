"""
Every number printed in the prose of the paper, as a LaTeX macro computed from the result files.

``numbers.tex`` defines one ``\\num<Name>`` macro per number; the prose uses the macros, so no number
in the text is typed by hand. ``NUMBERS.md`` lists, for each macro that the paper uses, the value as
printed, the section(s) that use it, the source file and the key it was computed from. The macros are
built from ``paper_tables.Results`` (the same loaders as the tables) and from the run JSONs in
``results/granularity/`` and ``results/mechanism/``, ``data/splits/*_stats.json``,
``results/granularity/taxonomy.json`` and the preprocessing constants.

Usage (repository root):
    python -m experiments.figures.paper_numbers --paper "<paper dir>"

``--paper`` is the directory with ``main.tex``; numbers.tex and NUMBERS.md are written there. The
script fails if ``main.tex`` uses a macro that is not defined, and reports the defined macros that
the text does not use (they are left out of NUMBERS.md).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from acw.data.preprocessing import config as prep
from acw.settings import ROOT, SEED
from experiments.figures import paper_tables as pt
from experiments.granularity import common as C

GTE, TC = pt.GTE, pt.TC
DOM = {"Music": "music", "Movies": "movies", "Games": "games"}
GRAN = pt.GRAN
GL = "results/granularity/"
ML = "results/mechanism/"
NUMS: dict[str, tuple[str, str, str]] = {}  # name -> (printed value, source, key)


# ───────────────────────────────── formatting ──────────────────────────────────


def f(x: float, nd: int) -> str:
    return f"{x:.{nd}f}"


def th(n: int) -> str:
    return f"{int(n):,}".replace(",", "{,}")


def rel(x: float) -> str:
    """Signed percent of a relative gain (fraction)."""
    return pt.signed(100 * x, 0, r"\%")


def sg3(x: float) -> str:
    """Signed difference in units of 1e-3 (unit stated in the text)."""
    return pt.signed(1e3 * x, 2)


def u3(x: float) -> str:
    """Signed difference with its unit."""
    return f"${'+' if x >= 0 else '-'}{abs(1e3 * x):.2f}\\times10^{{-3}}$"


def u3ci(lo: float, hi: float) -> str:
    s = lambda v: f"{'+' if v >= 0 else '-'}{abs(1e3 * v):.2f}"  # noqa: E731
    return f"$[{s(lo)},{s(hi)}]\\times10^{{-3}}$"


def pv(p: float) -> str:
    """A p-value as a math expression."""
    return "$p<10^{-3}$" if p < 1e-3 else f"$p={p:.3f}$"


def N(name: str, value: str, source: str, key: str) -> None:
    assert name not in NUMS, name
    NUMS[name] = (value, source, key)


# ────────────────────────────────── accessors ──────────────────────────────────


class A:
    """Thin accessors over the result files."""

    def __init__(self) -> None:
        self.R = pt.Results()
        self.run = {d: pt.run_json(d) for d in DOM.values()}
        self.mech = {d: pt.mech_json(d) for d in DOM.values()}
        self.splits = {
            d: json.loads((ROOT / "data" / "splits" / f"{d}_stats.json").read_text())
            for d in DOM.values()
        }
        self.hyb = {d: pt.hybrid_json(d) for d in DOM.values()}
        self.tax = json.loads((GRAN / "taxonomy.json").read_text())["domains"]

    def S(self, d, cond, g="exact", k="ndcg10", enc=GTE, text=TC) -> float:
        return float(self.R.summary.loc[(d, enc, text, g, cond)][k])

    def VR(self, d, cond, enc=GTE, text=TC, gt="exact", metric="ndcg10"):
        return pt._stat_row(self.R.vs_raw, domain=d, encoder=enc, text=text, gt=gt, metric=metric,
                            condition=cond)  # fmt: skip

    def Q(self, d, cond, enc=GTE, text=TC, k="ndcg10") -> float:
        return float(self.R.qn.loc[(d, enc, text, cond)][k])

    def QC(self, d, test, enc=GTE, text=TC):
        return pt.qn_contrast(self.R, d, enc, text, test)

    def QV(self, d, cond, enc=GTE, text=TC):
        return pt._stat_row(self.R.qn_stats, domain=d, encoder=enc, text=text, metric="ndcg10",
                            family="vs_raw", test=f"vs_raw:{cond}")  # fmt: skip

    def tally(self, test, family="contrast", metric="ndcg10", text=TC):
        r = pt._stat_row(self.R.qn_tally, domain="music", text=text, family=family, test=test,
                         metric=metric, encoders="verified")  # fmt: skip
        return int(r["better"]), int(r["ns"]), int(r["worse"])


def build() -> dict:
    a = A()
    R = a.R
    D = DOM
    # ───────────── setup and sizes ─────────────
    N(
        "VerifiedEncoders",
        str(len(pt.VERIFIED_MUSIC)),
        "paper_tables.ENCODERS",
        "verified encoders (docs/EMBEDDINGS.md)",
    )
    N("MinRating", f(prep.MIN_RATING, 0), "src/acw/data/preprocessing/config.py", "MIN_RATING")
    N(
        "KCore",
        str(prep.MIN_INTERACTIONS),
        "src/acw/data/preprocessing/config.py",
        "MIN_INTERACTIONS",
    )
    N("MinHistory", str(prep.MIN_HISTORY), "src/acw/data/preprocessing/config.py", "MIN_HISTORY")
    N("MinTextLen", str(prep.MIN_TEXT_LEN), "src/acw/data/preprocessing/config.py", "MIN_TEXT_LEN")
    N("Seed", str(SEED), "src/acw/settings.py; data/splits/*_stats.json", "SEED; seed")
    fr = a.splits["music"]["fractions"]
    N("SplitFractions", "/".join(str(round(100 * fr[k])) for k in ("train", "val", "test")),
      "data/splits/music_stats.json", "fractions")  # fmt: skip
    N("NBoot", th(C.N_BOOT), "experiments/granularity/common.py", "N_BOOT")
    fam = lambda enc: int(R.vs_raw[(R.vs_raw["domain"] == "music") & (R.vs_raw["encoder"] == enc)  # noqa: E731
                                   & (R.vs_raw["text"] == TC) & (R.vs_raw["gt"] == "exact")
                                   & (R.vs_raw["metric"] == "ndcg10")]["p_holm_signflip"].notna().sum())  # fmt: skip
    N(
        "FamilyGte",
        str(fam("gte")),
        GL + "stats_vs_raw.csv",
        "music/gte/titlecat exact ndcg10: rows with p_holm_signflip",
    )
    N(
        "FamilyOther",
        str(fam("bge-large")),
        GL + "stats_vs_raw.csv",
        "music/bge-large/titlecat exact ndcg10: rows with p_holm_signflip",
    )
    for L, d in D.items():
        N(
            f"TestUsers{L}",
            th(a.run[d]["n_eval_users"]["test"]),
            GL + f"{d}_gte_titlecat.json",
            "n_eval_users.test",
        )
        N(
            f"Pooled{L}",
            f(a.run[d]["mean_books_pooled_eval"]["test"], 1),
            GL + f"{d}_gte_titlecat.json",
            "mean_books_pooled_eval.test",
        )
    for L in ("Movies", "Games"):
        d = D[L]
        N(f"TestUsers{L}All", th(a.splits[d]["n_test"]), f"data/splits/{d}_stats.json", "n_test")
        N(
            f"Dropped{L}",
            str(a.run[d]["n_dropped_no_profile"]["test"]),
            GL + f"{d}_gte_titlecat.json",
            "n_dropped_no_profile.test",
        )
    # exploratory sample overlap
    legacy = json.loads((ROOT / "data/processed/canonical_test_users.json").read_text())
    legacy = set(legacy)
    test_ids = set(json.loads((ROOT / "data/splits/music_test.json").read_text()))
    N("LegacyOverlapPct", f(100 * len(legacy & test_ids) / len(legacy), 0), "data/processed/canonical_test_users.json; data/splits/music_test.json",
      "share of the 2,000 exploratory users that are in the music test split")  # fmt: skip

    # ───────────── granularity sizes ─────────────
    rs = {d: a.run[d]["relevance_size"]["test"] for d in D.values()}
    N(
        "RelExactMaxPct",
        f(0.05, 2),
        GL + "*_gte_titlecat.json",
        f"relevance_size.test.exact.median_rel_pct_of_catalog max {max(rs[d]['exact']['median_rel_pct_of_catalog'] for d in rs):.4f} < 0.05",
    )
    levels = [(d, g) for d in rs for g in rs[d] if g != "exact"]
    N(
        "RelCatMinPct",
        f(min(rs[d][g]["median_rel_pct_of_catalog"] for d, g in levels), 0),
        GL + "*_gte_titlecat.json",
        "relevance_size.test.<d3,d2,d1>.median_rel_pct_of_catalog (min)",
    )
    N(
        "RelCatMaxPct",
        f(max(rs[d][g]["median_rel_pct_of_catalog"] for d, g in levels), 0),
        GL + "*_gte_titlecat.json",
        "relevance_size.test.<d3,d2,d1>.median_rel_pct_of_catalog (max)",
    )
    rnd = [a.run[d]["means"]["test"][g]["random"]["nDCG@10"] for d, g in levels]
    N(
        "RandCatMin",
        f(min(rnd), 2),
        GL + "*_gte_titlecat.json",
        "means.test.<d2,d1,d3>.random.nDCG@10 (min)",
    )
    N(
        "RandCatMax",
        f(max(rnd), 2),
        GL + "*_gte_titlecat.json",
        "means.test.<d2,d1,d3>.random.nDCG@10 (max)",
    )
    N(
        "RelExactMusic",
        f(rs["music"]["exact"]["median_rel_pct_of_catalog"], 3),
        GL + "music_gte_titlecat.json",
        "relevance_size.test.exact.median_rel_pct_of_catalog",
    )
    N(
        "RelDTwoMusic",
        f(rs["music"]["d2"]["median_rel_pct_of_catalog"], 0),
        GL + "music_gte_titlecat.json",
        "relevance_size.test.d2.median_rel_pct_of_catalog",
    )
    N(
        "RelDOneMusic",
        f(rs["music"]["d1"]["median_rel_pct_of_catalog"], 0),
        GL + "music_gte_titlecat.json",
        "relevance_size.test.d1.median_rel_pct_of_catalog",
    )
    N(
        "RelDOneGames",
        f(rs["games"]["d1"]["median_rel_pct_of_catalog"], 0),
        GL + "games_gte_titlecat.json",
        "relevance_size.test.d1.median_rel_pct_of_catalog",
    )
    promo = dict((k, v) for k, v in a.tax["music"]["positions"]["1"]["top"])[
        "Today's Deals in Music"
    ]
    N(
        "PromoMusic",
        f(100 * promo, 1),
        GL + "taxonomy.json",
        "music.positions.1.top: Today's Deals in Music share",
    )
    N(
        "NotDeptMovies",
        f(100 * (1 - a.tax["movies"]["position0"]["share_modal"]), 1),
        GL + "taxonomy.json",
        "movies.position0.share_modal (1 - share)",
    )

    # ───────────── item level, grid (Table 2) ─────────────
    for L, d in D.items():
        raw, zca = a.S(d, "raw"), a.S(d, "zca_profile")
        N(f"Raw{L}", f(raw, 4), GL + "summary_test.csv", f"{d}/gte/titlecat exact raw ndcg10")
        N(
            f"Zca{L}",
            f(zca, 4),
            GL + "summary_test.csv",
            f"{d}/gte/titlecat exact zca_profile ndcg10",
        )
        N(
            f"ZcaRel{L}",
            rel(zca / raw - 1),
            GL + "summary_test.csv",
            "zca_profile / raw - 1 (exact ndcg10)",
        )
        cat = a.S(d, "zca_catalog_a1.0")
        N(
            f"CatZcaRel{L}",
            rel(cat / raw - 1),
            GL + "summary_test.csv",
            "zca_catalog_a1.0 / raw - 1 (exact ndcg10)",
        )
        N(
            f"Both{L}",
            f(a.S(d, "zca_both_separate"), 4),
            GL + "summary_test.csv",
            "zca_both_separate exact ndcg10",
        )
        N(f"Pop{L}", f(a.S(d, "popularity"), 4), GL + "summary_test.csv", "popularity exact ndcg10")
        N(f"RawDtwo{L}", f(a.S(d, "raw", "d2"), 3), GL + "summary_test.csv", "raw d2 ndcg10")
        N(
            f"TopDtwo{L}",
            f(a.S(d, "top_category", "d2"), 3),
            GL + "summary_test.csv",
            "top_category d2 ndcg10",
        )
        N(
            f"TopDone{L}",
            f(a.S(d, "top_category", "d1"), 3),
            GL + "summary_test.csv",
            "top_category d1 ndcg10",
        )
    v = a.VR("music", "zca_profile")
    N(
        "ZcaDiffMusic",
        u3(v["mean_diff"]),
        GL + "stats_vs_raw.csv",
        "music gte exact ndcg10 zca_profile mean_diff",
    )
    N(
        "ZcaCiMusic",
        u3ci(v["ci95_low"], v["ci95_high"]),
        GL + "stats_vs_raw.csv",
        "ci95_low, ci95_high",
    )
    N(
        "ZcaPMusic",
        f"$p\\le{v['p_holm_signflip']:.4f}$",
        GL + "stats_vs_raw.csv",
        "music gte exact ndcg10 zca_profile p_holm_signflip",
    )
    N("ZcaNnzMusic", th(v["n_nonzero_diff"]), GL + "stats_vs_raw.csv", "n_nonzero_diff")
    cat_tested = pt.CATALOG_TESTED
    rows = [a.VR(d, c) for d in D.values() for c in cat_tested]
    N(
        "CatTestsGte",
        str(len(rows)),
        GL + "stats_vs_raw.csv",
        "gte/titlecat exact ndcg10, 10 catalog-side conditions x 3 domains",
    )
    N(
        "CatWorseGte",
        str(sum(1 for r in rows if r["p_holm_signflip"] < 0.05 and r["mean_diff"] < 0)),
        GL + "stats_vs_raw.csv",
        "count with p_holm_signflip < 0.05 and mean_diff < 0 (none better)",
    )
    assert not any(r["p_holm_signflip"] < 0.05 and r["mean_diff"] > 0 for r in rows)
    d2c = [a.S("music", c, "d2") for c in cat_tested]
    N(
        "CatDtwoMinMusic",
        f(min(d2c), 3),
        GL + "summary_test.csv",
        "music d2 ndcg10: min over the 10 catalog-side conditions",
    )
    N(
        "CatDtwoMaxMusic",
        f(max(d2c), 3),
        GL + "summary_test.csv",
        "music d2 ndcg10: max over the 10 catalog-side conditions",
    )
    for L in ("Music", "Movies"):
        d = D[L]
        N(
            f"CatZcaDtwo{L}",
            f(a.S(d, "zca_catalog_a1.0", "d2"), 3),
            GL + "summary_test.csv",
            "zca_catalog_a1.0 d2 ndcg10",
        )
        N(
            f"CatZcaDtwoRel{L}",
            rel(a.S(d, "zca_catalog_a1.0", "d2") / a.S(d, "raw", "d2") - 1),
            GL + "summary_test.csv",
            "zca_catalog_a1.0 / raw - 1 (d2)",
        )
    N(
        "ZcaDtwoMusic",
        f(a.S("music", "zca_profile", "d2"), 3),
        GL + "summary_test.csv",
        "zca_profile d2 ndcg10",
    )
    N(
        "ZcaDtwoRelMusic",
        rel(a.S("music", "zca_profile", "d2") / a.S("music", "raw", "d2") - 1),
        GL + "summary_test.csv",
        "zca_profile / raw - 1 (d2)",
    )
    N(
        "ZcaDtwoMovies",
        f(a.S("movies", "zca_profile", "d2"), 3),
        GL + "summary_test.csv",
        "zca_profile d2 ndcg10",
    )
    N(
        "RandDtwoGames",
        f(a.run["games"]["means"]["test"]["d2"]["random"]["nDCG@10"], 3),
        GL + "games_gte_titlecat.json",
        "means.test.d2.random.nDCG@10",
    )
    N(
        "RandDoneGames",
        f(a.run["games"]["means"]["test"]["d1"]["random"]["nDCG@10"], 3),
        GL + "games_gte_titlecat.json",
        "means.test.d1.random.nDCG@10",
    )
    N(
        "PcaGames",
        f(a.S("games", "pca_catalog"), 4),
        GL + "summary_test.csv",
        "games pca_catalog exact ndcg10",
    )
    # constant list vs corrections at d1 and d2 (GTE: 3 domains x raw + 13 corrections; verified)
    top = R.vs_top[R.vs_top["condition"] != "pca_catalog"]
    g = top[
        (top["text"] == TC)
        & (top["encoder"] == GTE)
        & (top["metric"] == "ndcg10")
        & top["gt"].isin(["d1", "d2"])
    ]
    N(
        "TopCatTestsGte",
        str(len(g)),
        GL + "stats_vs_top_category.csv",
        "gte/titlecat ndcg10 d1 and d2: raw and the corrections (no PCA)",
    )
    N(
        "TopCatBeatsGte",
        str(int(((g["p_holm_signflip"] < 0.05) & (g["mean_diff"] < 0)).sum())),
        GL + "stats_vs_top_category.csv",
        "count significantly worse than the top-category list",
    )
    g = top[
        (top["text"] == TC) & top["verified"] & (top["metric"] == "ndcg10") & (top["gt"] == "d2")
    ]
    N(
        "TopCatTestsVer",
        str(len(g)),
        GL + "stats_vs_top_category.csv",
        "verified title+categories runs, d2, raw and the corrections",
    )
    N(
        "TopCatBeatsVer",
        str(int(((g["p_holm_signflip"] < 0.05) & (g["mean_diff"] < 0)).sum())),
        GL + "stats_vs_top_category.csv",
        "count significantly worse than the top-category list",
    )

    # ───────────── shuffled profiles ─────────────
    cs = R.ctl
    sh = lambda d, c: float(cs.loc[(d, TC, c)]["d2_ndcg10"])  # noqa: E731
    N(
        "ShufCatD",
        f(sh("music", "shuffled_zca_catalog_a1.0"), 3),
        GL + "summary_controls.csv",
        "music shuffled_zca_catalog_a1.0 d2_ndcg10",
    )
    N(
        "ShuffledCatD",
        f(100 * sh("music", "shuffled_zca_catalog_a1.0") / sh("music", "zca_catalog_a1.0"), 0),
        GL + "summary_controls.csv",
        "shuffled / real d2 (music, catalog ZCA)",
    )
    N(
        "ShufCatDMovies",
        f(100 * sh("movies", "shuffled_zca_catalog_a1.0") / sh("movies", "zca_catalog_a1.0"), 0),
        GL + "summary_controls.csv",
        "shuffled / real d2 (movies, catalog ZCA)",
    )
    N(
        "ShufCatDGames",
        f(100 * sh("games", "shuffled_zca_catalog_a1.0") / sh("games", "zca_catalog_a1.0"), 0),
        GL + "summary_controls.csv",
        "shuffled / real d2 (games, catalog ZCA)",
    )
    N(
        "ShufZcaD",
        f(
            100
            * sh("music", "shuffled_zca_train_users_a1.0")
            / sh("music", "zca_train_users_a1.0"),
            0,
        ),
        GL + "summary_controls.csv",
        "shuffled / real d2 (music, ZCA of the query)",
    )
    ct = R.ctl_stats
    cts = ct[(ct["metric"] == "ndcg10") & (ct["text"] == TC)]
    N(
        "ShuffleTests",
        str(int(((cts["p_holm_signflip"] < 0.05) & (cts["mean_diff"] < 0)).sum())),
        GL + "stats_controls.csv",
        "tests with Holm sign-flip p < 0.05, shuffled worse than real (of 18)",
    )
    assert len(cts) == 18
    mp = float(cts["p_holm_signflip"].max())
    N(
        "ShuffleMaxP",
        f"$p\\le{mp:.4f}$",
        GL + "stats_controls.csv",
        "max p_holm_signflip over the 18 tests",
    )
    N(
        "ShufZcaExactMusic",
        f(float(cs.loc[("music", TC, "shuffled_zca_train_users_a1.0")]["ndcg10"]), 4),
        GL + "summary_controls.csv",
        "music shuffled_zca_train_users_a1.0 ndcg10",
    )
    N(
        "ShufHitsReal",
        str(int(cs.loc[("music", TC, "zca_train_users_a1.0")]["n_hit10"])),
        GL + "summary_controls.csv",
        "music zca_train_users_a1.0 n_hit10",
    )
    N(
        "ShufHitsShuf",
        str(int(cs.loc[("music", TC, "shuffled_zca_train_users_a1.0")]["n_hit10"])),
        GL + "summary_controls.csv",
        "music shuffled_zca_train_users_a1.0 n_hit10",
    )
    for L in ("Movies", "Games"):
        d = D[L]
        N(
            f"ShufZcaHits{L}",
            str(int(cs.loc[(d, TC, "shuffled_zca_train_users_a1.0")]["n_hit10"])),
            GL + "summary_controls.csv",
            f"{d} shuffled_zca_train_users_a1.0 n_hit10",
        )
        N(
            f"ZcaHits{L}",
            th(cs.loc[(d, TC, "zca_train_users_a1.0")]["n_hit10"]),
            GL + "summary_controls.csv",
            f"{d} zca_train_users_a1.0 n_hit10",
        )

    # ───────────── 5.2 query normalisation ─────────────
    N(
        "QnZcaUMusic",
        f(a.Q("music", "zca_train_users_a1.0"), 4),
        GL + "summary_query_norm.csv",
        "music gte titlecat zca_train_users_a1.0 ndcg10",
    )
    N(
        "QnZcaCatMusic",
        f(a.Q("music", "zca_catalog_a1.0"), 4),
        GL + "summary_query_norm.csv",
        "music gte titlecat zca_catalog_a1.0 ndcg10",
    )
    r = a.QC("music", "a_zca_train_users_vs_zca_catalog")
    N(
        "AOneDiffMusic",
        u3(r["mean_diff"]),
        GL + "stats_query_norm.csv",
        "contrast a_zca_train_users_vs_zca_catalog, music: mean_diff",
    )
    N(
        "AOneCiMusic",
        u3ci(r["ci95_low"], r["ci95_high"]),
        GL + "stats_query_norm.csv",
        "ci95_low, ci95_high",
    )
    N(
        "AOnePMusic",
        pv(r["p_holm_signflip"])
        if r["p_holm_signflip"] >= 1e-3
        else f"$p\\le{r['p_holm_signflip']:.4f}$",
        GL + "stats_query_norm.csv",
        "p_holm_signflip",
    )
    N("AOneNnzMusic", th(r["n_nonzero_diff"]), GL + "stats_query_norm.csv", "n_nonzero_diff")
    t = lambda x: " / ".join(str(v) for v in x)  # noqa: E731
    N(
        "AOneTally",
        t(a.tally("a_zca_train_users_vs_zca_catalog")),
        GL + "tally_query_norm.csv",
        "contrast a_zca_train_users_vs_zca_catalog, music, verified: better / ns / worse",
    )
    N(
        "AFourTally",
        t(a.tally("a_selected_query_vs_selected_catalog")),
        GL + "tally_query_norm.csv",
        "contrast a_selected_query_vs_selected_catalog, music, verified",
    )
    N(
        "BTally",
        t(a.tally("b_zca_vs_center_train_users")),
        GL + "tally_query_norm.csv",
        "contrast b_zca_vs_center_train_users, music, verified",
    )
    N(
        "BTallyBetter",
        str(a.tally("b_zca_vs_center_train_users")[0]),
        GL + "tally_query_norm.csv",
        "same, better",
    )
    N(
        "CTally",
        t(a.tally("c_both_sides_vs_query_only")),
        GL + "tally_query_norm.csv",
        "contrast c_both_sides_vs_query_only, music, verified",
    )
    sel = {d: a.QC(d, "a_selected_query_vs_selected_catalog") for d in D.values()}

    def alpha_of(c):
        m = re.search(r"_a([\d.]+)$", c)
        return f"{float(m.group(1)):g}" if m else None

    N(
        "AFourAlphaMusic",
        alpha_of(sel["music"]["condition"]),
        GL + "stats_query_norm.csv",
        "music a_selected: condition",
    )
    N(
        "AFourAlphaGames",
        alpha_of(sel["games"]["condition"]),
        GL + "stats_query_norm.csv",
        "games a_selected: condition",
    )
    assert (
        sel["movies"]["condition"] == "zcalw_train_users"
        and "train_users" in sel["music"]["reference"] + sel["games"]["reference"]
    )
    # centring and books shares
    shares = {}
    for L, d in D.items():
        raw, z = a.Q(d, "raw"), a.Q(d, "zca_train_users_a1.0")
        shares[L] = 100 * (a.Q(d, "center_train_users") - raw) / (z - raw)
        N(
            f"CenterShare{L}",
            f(shares[L], 0),
            GL + "summary_query_norm.csv",
            f"{d}: (center_train_users - raw) / (zca_train_users_a1.0 - raw) in percent",
        )
        N(
            f"BooksShare{L}",
            f(100 * (a.Q(d, "zca_books_catalog_a1.0") - raw) / (z - raw), 0),
            GL + "summary_query_norm.csv",
            f"{d}: (zca_books_catalog_a1.0 - raw) / (zca_train_users_a1.0 - raw) in percent",
        )
    N(
        "CenterShareMin",
        f(min(shares.values()), 0),
        GL + "summary_query_norm.csv",
        "min over the three domains",
    )
    N(
        "CenterShareMax",
        f(max(shares.values()), 0),
        GL + "summary_query_norm.csv",
        "max over the three domains",
    )
    rb = {d: a.QC(d, "b_zca_vs_center_train_users") for d in D.values()}
    N(
        "BMoviesDiff",
        u3(rb["movies"]["mean_diff"]),
        GL + "stats_query_norm.csv",
        "contrast b, movies: mean_diff",
    )
    N(
        "BMoviesP",
        pv(rb["movies"]["p_holm_signflip"]),
        GL + "stats_query_norm.csv",
        "contrast b, movies: p_holm_signflip",
    )
    N(
        "BMusicP",
        pv(rb["music"]["p_holm_signflip"]),
        GL + "stats_query_norm.csv",
        "contrast b, music: p_holm_signflip",
    )
    N(
        "BGamesP",
        pv(rb["games"]["p_holm_signflip"]),
        GL + "stats_query_norm.csv",
        "contrast b, games: p_holm_signflip",
    )
    rd1 = {d: a.QC(d, "post_hoc:zca_target_catalog_vs_raw") for d in D.values()}
    N(
        "DOneDiffMusic",
        u3(rd1["music"]["mean_diff"]),
        GL + "stats_query_norm.csv",
        "post_hoc zca_target_catalog_vs_raw, music: mean_diff",
    )
    N(
        "DOneCiMusic",
        u3ci(rd1["music"]["ci95_low"], rd1["music"]["ci95_high"]),
        GL + "stats_query_norm.csv",
        "ci95_low, ci95_high",
    )
    N(
        "DOneMusicP",
        pv(rd1["music"]["p_holm_signflip"]),
        GL + "stats_query_norm.csv",
        "p_holm_signflip",
    )
    N(
        "DOneMoviesDiff",
        u3(rd1["movies"]["mean_diff"]),
        GL + "stats_query_norm.csv",
        "post_hoc ..., movies: mean_diff (p_holm_signflip < 0.001)",
    )
    N(
        "DOneGamesDiff",
        u3(rd1["games"]["mean_diff"]),
        GL + "stats_query_norm.csv",
        "post_hoc ..., games: mean_diff (p_holm_signflip < 0.001)",
    )
    assert rd1["movies"]["p_holm_signflip"] < 0.05 and rd1["games"]["p_holm_signflip"] < 0.05
    tb = a.tally("post_hoc:zca_target_catalog_vs_raw", "post_hoc")
    N(
        "DOneBetter",
        str(tb[0]),
        GL + "tally_query_norm.csv",
        "post_hoc zca_target_catalog_vs_raw, music, verified: better",
    )
    N("DOneNs", str(tb[1]), GL + "tally_query_norm.csv", "same, ns")
    N("DOneWorse", str(tb[2]), GL + "tally_query_norm.csv", "same, worse")
    rc = {d: a.QC(d, "c_both_sides_vs_query_only") for d in D.values()}
    for L, d in D.items():
        N(
            f"C{L}Diff",
            u3(rc[d]["mean_diff"]),
            GL + "stats_query_norm.csv",
            f"contrast c, {d}: mean_diff",
        )
        N(
            f"C{L}P",
            pv(rc[d]["p_holm_signflip"]),
            GL + "stats_query_norm.csv",
            f"contrast c, {d}: p_holm_signflip",
        )
    rcr = a.R.qn_stats
    rcr = rcr[
        (rcr["test"] == "c_both_sides_vs_query_only")
        & (rcr["metric"] == "r100")
        & (rcr["text"] == TC)
        & (rcr["domain"] == "music")
    ]
    N(
        "CRHundredWorse",
        str(int(((rcr["p_holm_signflip"] < 0.05) & (rcr["mean_diff"] < 0)).sum())),
        GL + "stats_query_norm.csv",
        "contrast c_both_sides_vs_query_only, metric r100, music, verified encoders: significantly worse (of 8)",
    )
    N(
        "HalfShareMusic",
        f(
            100
            * (a.Q("music", "zca_train_users_a0.5") - a.Q("music", "raw"))
            / (a.Q("music", "zca_train_users_a1.0") - a.Q("music", "raw")),
            0,
        ),
        GL + "summary_query_norm.csv",
        "music: (zca_train_users_a0.5 - raw) / (zca_train_users_a1.0 - raw) in percent",
    )
    N(
        "LwMusic",
        f(a.Q("music", "zcalw_train_users"), 5),
        GL + "summary_query_norm.csv",
        "music zcalw_train_users ndcg10",
    )
    N(
        "ZcaMusicThree",
        f(a.Q("music", "zca_train_users_a1.0"), 5),
        GL + "summary_query_norm.csv",
        "music zca_train_users_a1.0 ndcg10",
    )
    ab = [a.Q("music", f"abtt_train_users_D{k}") for k in (1, 3, 10)]
    N(
        "AbttMusicMin",
        f(min(ab), 5),
        GL + "summary_query_norm.csv",
        "music abtt_train_users_D{1,3,10} ndcg10 (min)",
    )
    N(
        "AbttMusicMax",
        f(max(ab), 5),
        GL + "summary_query_norm.csv",
        "music abtt_train_users_D{1,3,10} ndcg10 (max)",
    )
    N(
        "CenterMusicThree",
        f(a.Q("music", "center_train_users"), 5),
        GL + "summary_query_norm.csv",
        "music center_train_users ndcg10",
    )
    N(
        "CatCenterRelMusic",
        rel(a.S("music", "center_catalog") / a.S("music", "raw") - 1),
        GL + "summary_test.csv",
        "center_catalog / raw - 1 (music exact)",
    )
    # encoders
    gains, cat_worse = [], 0
    for enc in pt.VERIFIED_MUSIC:
        gains.append(
            (a.Q("music", "zca_train_users_a1.0", enc) / a.Q("music", "raw", enc) - 1, enc)
        )
        r = a.QV("music", "zca_catalog_a1.0", enc)
        cat_worse += int(r["p_holm_signflip"] < 0.05 and r["mean_diff"] < 0)
    N(
        "EncGainMin",
        f(100 * min(gains)[0], 0),
        GL + "summary_query_norm.csv",
        f"music, verified encoders: zca_train_users_a1.0 / raw - 1 (min: {min(gains)[1]})",
    )
    N(
        "EncGainMax",
        f(100 * max(gains)[0], 0),
        GL + "summary_query_norm.csv",
        f"music, verified encoders: zca_train_users_a1.0 / raw - 1 (max: {max(gains)[1]})",
    )
    N(
        "EncCatWorse",
        str(cat_worse),
        GL + "stats_query_norm.csv",
        "vs_raw:zca_catalog_a1.0, music, verified encoders with p_holm_signflip < 0.05 and mean_diff < 0",
    )
    raws = [a.Q("music", "raw", e) for e in pt.VERIFIED_MUSIC]
    N(
        "RawSpread",
        f(max(raws) / min(raws), 1),
        GL + "summary_query_norm.csv",
        "music, verified encoders: max raw ndcg10 / min raw ndcg10",
    )
    # tallies by side
    cnt = lambda df, conds, ver, gt: pt.tally_counts(df, conds, ver, gt)  # noqa: E731
    cb = cnt(R.vs_raw, pt.CATALOG_TESTED, True, "exact")
    cu = cnt(R.vs_raw, pt.CATALOG_TESTED, False, "exact")
    N(
        "TallyCatBetterVer",
        str(cb[0]),
        GL + "stats_vs_raw.csv",
        "catalog-side conditions, title+categories runs, verified, exact ndcg10: better",
    )
    N("TallyCatWorseVer", str(cb[2]), GL + "stats_vs_raw.csv", "same: worse")
    N("TallyCatNVer", str(cb[3]), GL + "stats_vs_raw.csv", "same: tests")
    N("TallyCatBetterUnv", str(cu[0]), GL + "stats_vs_raw.csv", "same, unverified: better")
    N("TallyCatNUnv", str(cu[3]), GL + "stats_vs_raw.csv", "same, unverified: tests")
    N(
        "TallyCatBetterAll",
        str(cb[0] + cu[0]),
        GL + "stats_vs_raw.csv",
        "verified + unverified better",
    )
    N("TallyCatNAll", str(cb[3] + cu[3]), GL + "stats_vs_raw.csv", "verified + unverified tests")
    N(
        "TallyCatDTwoBetter",
        str(cnt(R.vs_raw, pt.CATALOG_TESTED, True, "d2")[0]),
        GL + "stats_vs_raw.csv",
        "catalog side, verified, d2 ndcg10: better",
    )
    assert (
        cnt(R.vs_top, pt.CATALOG_TESTED, True, "d2")[0] == 0
        and cnt(R.vs_top, pt.CATALOG_TESTED, False, "d2")[0] == 0
    )
    N(
        "TallyQVerBetter",
        str(cnt(R.vs_raw, ["zca_profile"], True, "exact")[0]),
        GL + "stats_vs_raw.csv",
        "zca_profile, verified, exact ndcg10: better (of 12 runs, none worse)",
    )
    v = R.vs_raw[(R.vs_raw["text"] == TC) & (R.vs_raw["gt"] == "exact") & (R.vs_raw["metric"] == "ndcg10")
                 & R.vs_raw["condition"].isin(pt.CATALOG_TESTED)]  # fmt: skip
    pos = v[(v["p_holm_signflip"] < 0.05) & (v["mean_diff"] > 0) & v["verified"]]
    N(
        "PosArctic",
        str(
            int(((pos["domain"] == "music") & (pos["encoder"] == "snowflake-arctic-embed-l")).sum())
        ),
        GL + "stats_vs_raw.csv",
        "verified positives: music, snowflake-arctic-embed-l",
    )
    N(
        "PosBge",
        str(int(((pos["domain"] == "movies") & (pos["encoder"] == "bge-large")).sum())),
        GL + "stats_vs_raw.csv",
        "verified positives: movies, bge-large",
    )
    assert len(pos) == cb[0]

    # ───────────── 5.3 text format ─────────────
    for tag, text in (("Title", "title"), ("Tc", "titlecat"), ("Desc", "desc")):
        raw = a.Q("music", "raw", GTE, text)
        N(
            f"TextRaw{tag}" if tag != "Title" else "TextRawTitle",
            f(raw, 4),
            GL + "summary_query_norm.csv",
            f"music gte {text} raw ndcg10",
        )
        if tag != "Tc":
            N(
                f"TextZcaRel{tag}",
                f(100 * (a.Q("music", "zca_train_users_a1.0", GTE, text) / raw - 1), 0),
                GL + "summary_query_norm.csv",
                f"music gte {text}: zca_train_users_a1.0 / raw - 1 in percent",
            )
        d2r, d2c = (
            a.Q("music", "raw", GTE, text, "d2_ndcg10"),
            a.Q("music", "zca_catalog_a1.0", GTE, text, "d2_ndcg10"),
        )
        if tag != "Desc":
            N(
                f"TextRawDtwo{tag}",
                f(d2r, 3),
                GL + "summary_query_norm.csv",
                f"music gte {text} raw d2_ndcg10",
            )
            N(
                f"TextCatDtwo{tag}",
                f(d2c, 3),
                GL + "summary_query_norm.csv",
                f"music gte {text} zca_catalog_a1.0 d2_ndcg10",
            )
            N(
                f"TextCatDtwoGain{tag}",
                pt.signed(d2c - d2r, 3),
                GL + "summary_query_norm.csv",
                f"music gte {text}: zca_catalog_a1.0 - raw (d2_ndcg10)",
            )
    pa = max(
        float(a.QC("music", "a_zca_train_users_vs_zca_catalog", GTE, tx)["p_holm_signflip"])
        for tx in ("title", "titlecat", "desc")
    )
    N(
        "TextAOneMaxP",
        f"$p\\le{pa:.4f}$",
        GL + "stats_query_norm.csv",
        "contrast a_zca_train_users_vs_zca_catalog, music, text title/titlecat/desc: max p_holm_signflip",
    )
    N(
        "ZcaRelMusicPlain",
        f(100 * (a.S("music", "zca_profile") / a.S("music", "raw") - 1), 0),
        GL + "summary_test.csv",
        "music zca_profile / raw - 1 in percent",
    )

    # ───────────── 5.4 lexical, cold ─────────────
    ls = R.lex_stats
    lx = ls[
        (ls["family"] == "lexical_vs_content") & (ls["metric"] == "ndcg10") & (ls["text"] == TC)
    ]
    cen = lx[lx["test"].str.endswith("_vs_center_train_users")]
    N(
        "LexMinPCentre",
        pv(float(cen["p_holm_signflip"].min())),
        GL + "stats_lexical.csv",
        "min p_holm_signflip over the 12 tests of the four lexical scorers against center_train_users",
    )
    r = cen[cen["test"] == "bm25_title_vs_center_train_users"]
    r = r[r["domain"] == "movies"].iloc[0]
    assert r["p_holm_signflip"] == cen["p_holm_signflip"].min()
    N(
        "LexMoviesBmDiff",
        u3(r["mean_diff"]),
        GL + "stats_lexical.csv",
        "bm25_title_vs_center_train_users, movies: mean_diff",
    )
    zc = lx[lx["test"].str.endswith("_vs_zca_train_users_a1.0")]
    mv = zc[(zc["domain"] == "movies") & zc["test"].str.startswith("idf_overlap")]
    N(
        "LexMoviesIdfP",
        pv(float(mv["p_holm_signflip"].max())),
        GL + "stats_lexical.csv",
        "idf_overlap_{title,titlecat}_vs_zca_train_users_a1.0, movies: max p_holm_signflip",
    )
    mg = zc[(zc["domain"].isin(["music", "games"])) & zc["test"].str.startswith("idf_overlap")]
    N(
        "LexIdfMinPZca",
        pv(float(mg["p_holm_signflip"].min())),
        GL + "stats_lexical.csv",
        "idf_overlap_* vs zca_train_users_a1.0, music and games: min p_holm_signflip",
    )
    assert (zc[zc["test"].str.startswith("bm25")]["p_holm_signflip"] < 0.05).all() and (
        mv["p_holm_signflip"] < 0.05
    ).all()
    assert (mg["p_holm_signflip"] >= 0.05).all() and (cen["p_holm_signflip"] >= 0.05).all()
    # long music text (desc): BM25 / IDF overlap against ZCA(U), centred, raw and popularity
    dq = lambda c: float(R.qn.loc[("music", GTE, "desc", c)]["ndcg10"])  # noqa: E731
    for lab, cond in (("Zca", "zca_train_users_a1.0"), ("Ctr", "center_train_users"), ("Raw", "raw"), ("Pop", "popularity")):  # fmt: skip
        N(f"LexDesc{lab}Ndcg", f(dq(cond), 4), GL + "summary_query_norm.csv", f"music gte desc {cond} ndcg10")  # fmt: skip
    for ab, lc in (("Bm", "bm25_desc"), ("Idf", "idf_overlap_desc")):
        src = GL + "stats_lexical.csv"
        N(f"LexDesc{ab}Ndcg", f(float(R.lex.loc[("music", "desc", lc)]["ndcg10"]), 4), GL + "summary_lexical.csv", f"music desc {lc} ndcg10")  # fmt: skip
        for lab, cond in (("Zca", "zca_train_users_a1.0"), ("Ctr", "center_train_users"), ("Raw", "raw"), ("Pop", "popularity")):  # fmt: skip
            t = ls[(ls["domain"] == "music") & (ls["text"] == "desc") & (ls["metric"] == "ndcg10") & (ls["family"] == "lexical_vs_content") & (ls["test"] == f"{lc}_vs_{cond}")].iloc[0]  # fmt: skip
            k = f"{lc}_vs_{cond}"
            N(f"LexDesc{ab}{lab}Diff", u3(t["mean_diff"]), src, f"{k}: mean_diff")
            N(f"LexDesc{ab}{lab}CI", u3ci(t["ci95_low"], t["ci95_high"]), src, f"{k}: ci95")
            N(f"LexDesc{ab}{lab}P", pv(float(t["p_holm_signflip"])), src, f"{k}: p_holm_signflip")  # fmt: skip
            N(f"LexDesc{ab}{lab}Nnz", th(int(t["n_nonzero_diff"])), src, f"{k}: n_nonzero_diff")
    for L, d in D.items():
        N(
            f"LexStrict{L}",
            f(float(R.lex.loc[(d, TC, "strict_idf_h2_window")]["ndcg10"]), 4),
            GL + "summary_lexical.csv",
            f"{d} strict_idf_h2_window ndcg10 (tokens of at least four letters, ties by train popularity)",
        )
        n = a.Q(d, "raw", k="n_users")
        c = a.Q(d, "raw", k="cold_n_users")
        N(f"ColdUsers{L}", th(c), GL + "summary_query_norm.csv", f"{d} raw cold_n_users")
        N(
            f"ColdUsersPct{L}",
            f(100 * c / n, 0),
            GL + "summary_query_norm.csv",
            f"{d} cold_n_users / n_users in percent",
        )
        N(
            f"ColdZca{L}",
            f(1e3 * a.Q(d, "zca_train_users_a1.0", k="cold_ndcg10"), 2),
            GL + "summary_query_norm.csv",
            f"{d} zca_train_users_a1.0 cold_ndcg10 x 1000",
        )
        N(
            f"ColdHitsZca{L}",
            str(int(a.Q(d, "zca_train_users_a1.0", k="cold_n_hit10"))),
            GL + "summary_query_norm.csv",
            f"{d} zca_train_users_a1.0 cold_n_hit10",
        )

    # ───────────── 5.5 logs ─────────────
    vp = R.vs_pop[R.vs_pop["condition"] != "pca_catalog"]  # PCA is outside the counts
    g = vp[
        (vp["text"] == TC)
        & (vp["encoder"] == GTE)
        & (vp["metric"] == "ndcg10")
        & (vp["gt"] == "exact")
    ]
    N(
        "PopTestsGte",
        str(len(g)),
        GL + "stats_vs_popularity.csv",
        "gte/titlecat exact ndcg10: raw and the corrections (no PCA), 3 domains",
    )
    N(
        "PopWorseGte",
        str(int(((g["p_holm_signflip"] < 0.05) & (g["mean_diff"] < 0)).sum())),
        GL + "stats_vs_popularity.csv",
        "count significantly worse than popularity",
    )
    N(
        "PopMaxP",
        f"$p\\le{float(g['p_holm_signflip'].max()):.4f}$",
        GL + "stats_vs_popularity.csv",
        "max p_holm_signflip",
    )
    gv = vp[
        (vp["text"] == TC) & vp["verified"] & (vp["metric"] == "ndcg10") & (vp["gt"] == "exact")
    ]
    N(
        "PopTestsVer",
        str(len(gv)),
        GL + "stats_vs_popularity.csv",
        "verified title+categories runs, exact ndcg10",
    )
    N(
        "PopWorseVer",
        str(int(((gv["p_holm_signflip"] < 0.05) & (gv["mean_diff"] < 0)).sum())),
        GL + "stats_vs_popularity.csv",
        "count significantly worse than popularity",
    )
    for L, d in D.items():
        sub = R.summary.loc[(d, GTE, TC, "exact")]
        best = max(
            (c for c in sub.index if c not in pt.BASELINES), key=lambda c: sub.loc[c, "ndcg10"]
        )
        t_ = R.baseline_test("popularity", d, GTE, TC, "exact", best)
        N(
            f"PopGap{L}",
            f(abs(1e3 * t_["mean_diff"]), 1),
            GL + "stats_vs_popularity.csv",
            f"{d}: best grid condition by mean exact ndcg10 ({best}): |mean_diff| x 1000",
        )
    hp = {}
    for L, d in D.items():
        n = a.Q(d, "raw", k="n_users")
        hp[L] = {
            c: a.Q(d, c, k="n_hit10") / n for c in ("raw", "zca_train_users_a1.0", "popularity")
        }
        un = a.mech[d]["D"]["n_users_nonzero_ndcg"]
        N(f"UsersHit{L}", th(un), ML + f"{d}_gte.json", "D.n_users_nonzero_ndcg")
        N(
            f"HitPct{L}",
            f(100 * un / n, 1),
            ML + f"{d}_gte.json",
            "D.n_users_nonzero_ndcg / D.n_users",
        )
    N(
        "HitPctMin",
        min((NUMS[f"HitPct{L}"][0] for L in D), key=float),
        ML + "*_gte.json",
        "min over domains",
    )
    N(
        "HitPctMax",
        max((NUMS[f"HitPct{L}"][0] for L in D), key=float),
        ML + "*_gte.json",
        "max over domains",
    )
    N(
        "PopHitPctMin",
        f(100 * min(hp[L]["popularity"] for L in D), 1),
        GL + "summary_query_norm.csv",
        "popularity n_hit10 / n_users, min over domains",
    )
    N(
        "PopHitPctMax",
        f(100 * max(hp[L]["popularity"] for L in D), 1),
        GL + "summary_query_norm.csv",
        "popularity n_hit10 / n_users, max over domains",
    )
    ks = R.knn_stats
    ks_sel = {
        json.dumps(
            json.loads((GRAN / f"knn_{d}_gte.json").read_text())["selected_k"], sort_keys=True
        )
        for d in D.values()
    }
    kv = {
        v
        for d in D.values()
        for v in json.loads((GRAN / f"knn_{d}_gte.json").read_text())["selected_k"].values()
    }
    del ks_sel, kv
    for L, d in D.items():
        for v, tag in (("raw", "Raw"), ("center_train_users", "Ctr")):
            N(f"KnnK{tag}{L}", str(pt.knn_sel(d)[v]), GL + f"knn_{d}_gte.json", f"selected_k[{v}]")
        jd = json.loads((GRAN / f"knn_{d}_gte.json").read_text())
        v500 = jd["val_k_curve"]["knn_center_train_users_k500"]["nDCG@10"]
        vsel = jd["val_k_curve"][f"knn_center_train_users_k{pt.knn_sel(d)['center_train_users']}"][
            "nDCG@10"
        ]
        N(
            f"KnnValKFiveHundred{L}",
            f(v500, 4),
            GL + f"knn_{d}_gte.json",
            "val_k_curve centred k500 nDCG@10",
        )
        N(f"KnnValKSel{L}", f(vsel, 4), GL + f"knn_{d}_gte.json", "val_k_curve centred selected k nDCG@10")  # fmt: skip
    for L, d in D.items():
        N(
            f"Knn{L}",
            f(float(R.knn.loc[(d, TC, pt.rk(d, "knn_center_train_users_kSEL"))]["ndcg10"]), 4),
            GL + "summary_knn.csv",
            f"{d} knn_center_train_users_k(selected) ndcg10",
        )
    kv = ks[(ks["family"] == "knn_vs_references") & (ks["metric"] == "ndcg10") & (ks["text"] == TC)]
    kc_ = kv[
        kv["test"].str.contains("_vs_(?:raw|zca_train_users_a1.0|center_train_users)$", regex=True)
    ]
    assert len(kc_) == 18 and (kc_["p_holm_signflip"] < 0.05).all() and (kc_["mean_diff"] > 0).all()
    N(
        "KnnVsContentP",
        f"$p\\le{float(kc_['p_holm_signflip'].max()):.4f}$",
        GL + "stats_knn.csv",
        "knn_{raw,center}_k(selected) vs raw, center_train_users, zca_train_users_a1.0, 3 domains (18 tests): max p_holm_signflip",
    )
    tm = pt._test(ks, "movies", pt.rk("movies", "knn_center_train_users_kSEL_vs_popularity"))
    N(
        "KnnMoviesPopP",
        pv(tm["p_holm_signflip"]),
        GL + "stats_knn.csv",
        "movies knn_center_train_users_k(selected)_vs_popularity: p_holm_signflip",
    )
    N(
        "KnnMoviesPopDiff",
        u3(tm["mean_diff"]),
        GL + "stats_knn.csv",
        "movies knn_center_train_users_k(selected)_vs_popularity: mean_diff",
    )
    tr = pt._test(ks, "music", pt.rk("music", "knn_raw_kSEL_vs_popularity"))
    N(
        "KnnMusicPopDiff",
        u3(tr["mean_diff"]),
        GL + "stats_knn.csv",
        "music knn_raw_k(selected)_vs_popularity: mean_diff",
    )
    N("KnnMusicPopP", pv(tr["p_holm_signflip"]), GL + "stats_knn.csv", "p_holm_signflip")
    N("KnnMusicPopPW", pv(tr["p_holm_wilcoxon"]), GL + "stats_knn.csv", "p_holm_wilcoxon")
    # hybrids with popularity
    hs = R.hyb_stats

    def hy(d, c, ref, k="p_holm_signflip"):
        return float(pt._stat_row(hs, domain=d, metric="ndcg10", condition=c, reference=ref)[k])

    for L, d in D.items():
        m = a.hyb[d]["test"]["exact"]
        N(
            f"Hyb{L}",
            f(m["hybrid_zca_profile"]["nDCG@10"], 4),
            GL + f"hybrid_{d}_gte.json",
            "test.exact.hybrid_zca_profile.nDCG@10",
        )
        N(
            f"HybRel{L}",
            rel(m["hybrid_zca_profile"]["nDCG@10"] / m["popularity"]["nDCG@10"] - 1),
            GL + f"hybrid_{d}_gte.json",
            "hybrid_zca_profile / popularity - 1 (test exact nDCG@10)",
        )
        N(
            f"HybP{L}",
            pv(hy(d, "hybrid_zca_profile", "popularity")),
            GL + "stats_hybrid.csv",
            f"{d} hybrid_zca_profile vs popularity: p_holm_signflip",
        )
        N(
            f"HybPW{L}",
            pv(hy(d, "hybrid_zca_profile", "popularity", "p_holm_wilcoxon")),
            GL + "stats_hybrid.csv",
            f"{d} hybrid_zca_profile vs popularity: p_holm_wilcoxon",
        )
        N(
            f"PopShare{L}",
            f(100 * m["popularity"]["nDCG@10"] / m["hybrid_zca_profile"]["nDCG@10"], 0),
            GL + f"hybrid_{d}_gte.json",
            "popularity / hybrid_zca_profile (test exact nDCG@10) in percent",
        )
        N(
            f"HybVsRaw{L}",
            sg3(hy(d, "hybrid_zca_profile", "hybrid_raw", "mean_diff")),
            GL + "stats_hybrid.csv",
            f"{d} hybrid_zca_profile vs hybrid_raw: mean_diff x 1000",
        )
        N(
            f"HybVsRawP{L}",
            pv(hy(d, "hybrid_zca_profile", "hybrid_raw")),
            GL + "stats_hybrid.csv",
            f"{d} hybrid_zca_profile vs hybrid_raw: p_holm_signflip",
        )
    for L in ("Music", "Games"):
        N(
            f"HybNnz{L}",
            th(hy(D[L], "hybrid_zca_profile", "popularity", "n_nonzero_diff")),
            GL + "stats_hybrid.csv",
            f"{D[L]} hybrid_zca_profile vs popularity: n_nonzero_diff",
        )
    over = hs[
        (hs["metric"].isin(["ndcg10", "r100"]))
        & hs["condition"].str.startswith("hybrid_")
        & (hs["reference"] == hs["condition"].str.replace("hybrid_", "", n=1))
    ]
    N(
        "HybOverPure",
        str(len(over)),
        GL + "stats_hybrid.csv",
        "hybrid_C vs C, both metrics, three domains (all with Holm sign-flip p < 0.05)",
    )
    assert (over["p_holm_signflip"] < 0.05).all()
    N(
        "HybOverPureP",
        f"$p\\le{float(over['p_holm_signflip'].max()):.4f}$",
        GL + "stats_hybrid.csv",
        "max p_holm_signflip of those tests",
    )
    N(
        "HybVsCatMax",
        f(
            max(
                1e3 * hy(d, "hybrid_zca_profile", "hybrid_zca_catalog_a1.0", "mean_diff")
                for d in D.values()
            ),
            2,
        ),
        GL + "stats_hybrid.csv",
        "hybrid_zca_profile vs hybrid_zca_catalog_a1.0: max mean_diff x 1000",
    )
    N(
        "HybOverPureMin",
        f(
            min(
                1e3 * hy(d, f"hybrid_{c}", c, "mean_diff")
                for d in D.values()
                for c in ("raw", "zca_profile", "zca_catalog_a1.0")
            ),
            2,
        ),
        GL + "stats_hybrid.csv",
        "min over 9 (domain, content) of hybrid_C vs C mean_diff x 1000 (nDCG@10)",
    )
    # hybrids with kNN
    kc = "hybrid_knn_center_train_users+zca_train_users_a1.0"
    for L, d in D.items():
        N(
            f"KnnHyb{L}",
            f(float(R.knn.loc[(d, TC, kc)]["ndcg10"]), 4),
            GL + "summary_knn.csv",
            f"{d} {kc} ndcg10",
        )
        t_ = pt._test(ks, d, f"{kc}_vs_" + pt.rk(d, "knn_center_train_users_kSEL"))
        N(
            f"KnnHybGain{L}",
            sg3(t_["mean_diff"]),
            GL + "stats_knn.csv",
            f"{d} {kc} vs knn_center_train_users_k(selected): mean_diff x 1000",
        )
        N(f"KnnHybP{L}", pv(t_["p_holm_signflip"]), GL + "stats_knn.csv", "p_holm_signflip")
        if L != "Games":
            N(f"KnnHybNnz{L}", th(t_["n_nonzero_diff"]), GL + "stats_knn.csv", "n_nonzero_diff")
    N(
        "KnnHybPopP",
        pv(pt._test(ks, "movies", f"{kc}_vs_popularity")["p_holm_signflip"]),
        GL + "stats_knn.csv",
        f"movies {kc} vs popularity: p_holm_signflip",
    )
    # discordant users, validation users, signal sizes
    nz = [a.VR(d, "zca_profile")["n_nonzero_diff"] for d in D.values()]
    N(
        "NnzMin",
        th(min(nz)),
        GL + "stats_vs_raw.csv",
        "zca_profile vs raw, exact ndcg10: n_nonzero_diff (min over domains)",
    )
    N("NnzMax", th(max(nz)), GL + "stats_vs_raw.csv", "same (max)")
    vu = [a.hyb[d]["n_eval_users"]["val"] for d in D.values()]
    N("ValUsersMin", th(min(vu)), GL + "hybrid_*_gte.json", "n_eval_users.val (min)")
    N("ValUsersMax", th(max(vu)), GL + "hybrid_*_gte.json", "n_eval_users.val (max)")

    # ───────────── 6 mechanism ─────────────
    M = a.mech
    N("MechUsers", th(M["music"]["B"]["n_users"]), ML + "music_gte.json", "B.n_users")
    N("MechPairs", th(M["music"]["B"]["n_pairs"]), ML + "music_gte.json", "B.n_pairs")
    N("MechNnItems", th(M["music"]["C"]["n_sample"]), ML + "music_gte.json", "C.n_sample")
    for L, d in D.items():
        m = M[d]
        Ap, Ac = m["A"]["profiles"], m["A"]["catalog"]
        B, E, Cc = m["B"]["conditions"], m["E"]["concentration"], m["C"]["conditions"]
        N(
            f"Collapse{L}",
            f(Ap["raw"]["mean_pairwise_cosine"], 2),
            ML + f"{d}_gte.json",
            "A.profiles.raw.mean_pairwise_cosine",
        )
        N(
            f"CatCos{L}",
            f(Ac["raw"]["mean_pairwise_cosine"], 2),
            ML + f"{d}_gte.json",
            "A.catalog.raw.mean_pairwise_cosine",
        )
        N(
            f"EffRank{L}",
            f(Ap["raw"]["effective_rank"], 0),
            ML + f"{d}_gte.json",
            "A.profiles.raw.effective_rank",
        )
        N(
            f"EffRankZca{L}",
            f(Ap["zca_profile"]["effective_rank"], 0),
            ML + f"{d}_gte.json",
            "A.profiles.zca_profile.effective_rank",
        )
        for tag, c in (
            ("Raw", "raw"),
            ("Cat", "zca_catalog_a1.0"),
            ("Ctr", "center_profile"),
            ("Prof", "zca_profile"),
        ):
            N(
                f"Cov{tag}{L}",
                f(B[c]["coverage_pct"], 1),
                ML + f"{d}_gte.json",
                f"B.conditions.{c}.coverage_pct",
            )
            N(
                f"Hub{tag}{L}",
                f(B[c]["hubness_skewness_k10"], 0 if tag in ("Raw", "Cat", "Ctr") else 1),
                ML + f"{d}_gte.json",
                f"B.conditions.{c}.hubness_skewness_k10",
            )
        N(
            f"GiniRaw{L}",
            f(B["raw"]["gini_exposure"], 3),
            ML + f"{d}_gte.json",
            "B.conditions.raw.gini_exposure",
        )
        for tag, c in (("Raw", "raw"), ("Cat", "zca_catalog_a1.0")):
            N(
                f"Jac{tag}{L}",
                f(B[c]["mean_jaccard"], 2),
                ML + f"{d}_gte.json",
                f"B.conditions.{c}.mean_jaccard",
            )
            N(
                f"Bw{tag}{L}",
                f(Cc[c]["between_within_ratio"], 2),
                ML + f"{d}_gte.json",
                f"C.conditions.{c}.between_within_ratio",
            )
            N(
                f"Nn{tag}{L}",
                f(Cc[c]["nn10_same_category"], 2),
                ML + f"{d}_gte.json",
                f"C.conditions.{c}.nn10_same_category",
            )
        for tag, c in (("Raw", "raw"), ("Cat", "zca_catalog_a1.0")):
            N(
                f"TopFive{tag}{L}",
                f(E[c]["share_top5_train_categories"], 2),
                ML + f"{d}_gte.json",
                f"E.concentration.{c}.share_top5_train_categories",
            )
        Dd = m["D"]
        N(f"Rho{L}", f(Dd["spearman_rho"], 3), ML + f"{d}_gte.json", "D.spearman_rho")
        N(
            f"RhoNz{L}",
            f(Dd["spearman_rho_nonzero"], 2),
            ML + f"{d}_gte.json",
            "D.spearman_rho_nonzero",
        )
        N(
            f"RhoJac{L}",
            f(
                m["E"]["jaccard_to_top_category"]["pooled_personalised"]["spearman_vs_d2_ndcg10"][
                    "rho"
                ],
                3,
            ),
            ML + f"{d}_gte.json",
            "E.jaccard_to_top_category.pooled_personalised.spearman_vs_d2_ndcg10.rho",
        )
    mu = M["music"]
    N(
        "PopPctRawMusic",
        f(mu["B"]["conditions"]["raw"]["mean_popularity_percentile"], 1),
        ML + "music_gte.json",
        "B.conditions.raw.mean_popularity_percentile",
    )
    N(
        "PopPctCtrMusic",
        f(mu["B"]["conditions"]["center_profile"]["mean_popularity_percentile"], 1),
        ML + "music_gte.json",
        "B.conditions.center_profile.mean_popularity_percentile",
    )
    N(
        "PopPctProfMusic",
        f(mu["B"]["conditions"]["zca_profile"]["mean_popularity_percentile"], 1),
        ML + "music_gte.json",
        "B.conditions.zca_profile.mean_popularity_percentile",
    )
    for tag, c in (
        ("Raw", "raw"),
        ("Cat", "zca_catalog_a1.0"),
        ("Ctr", "center_profile"),
        ("Prof", "zca_profile"),
    ):
        N(
            f"Ent{tag}Music",
            f(mu["E"]["concentration"][c]["entropy_bits"], 1),
            ML + "music_gte.json",
            f"E.concentration.{c}.entropy_bits",
        )
    N(
        "TopFiveProfMusic",
        f(mu["E"]["concentration"]["zca_profile"]["share_top5_train_categories"], 2),
        ML + "music_gte.json",
        "E.concentration.zca_profile.share_top5_train_categories",
    )
    enc = json.loads((GRAN.parent / "mechanism" / "music_encoders.json").read_text())
    N(
        "RhoEncCos",
        f(enc["spearman_raw_profile_mean_pairwise_cosine_vs_gain"]["rho"], 2),
        ML + "music_encoders.json",
        "spearman_raw_profile_mean_pairwise_cosine_vs_gain.rho",
    )
    N(
        "RhoEncCosP",
        f(enc["spearman_raw_profile_mean_pairwise_cosine_vs_gain"]["p"], 2),
        ML + "music_encoders.json",
        "... .p",
    )
    N(
        "RhoEncRank",
        f(enc["spearman_raw_profile_effective_rank_vs_gain"]["rho"], 2),
        ML + "music_encoders.json",
        "spearman_raw_profile_effective_rank_vs_gain.rho",
    )
    N(
        "RhoEncRankP",
        f(enc["spearman_raw_profile_effective_rank_vs_gain"]["p"], 2),
        ML + "music_encoders.json",
        "... .p",
    )
    round3(a)
    return NUMS


# ─────────────────────────── round 3: matrix, splits, overlap ───────────────────────────

MS_NAME = {
    "qU_vs_raw": "QU", "qCtr_vs_raw": "QCtr", "a1": "AOne", "a2": "ATwo", "a3": "AThree",
    "a4": "AFour", "b": "B", "c": "C", "d1": "DOne", "d2": "DTwo", "itemsU_vs_raw": "ItemsU",
    "itemsB_vs_raw": "ItemsB", "itemsC_vs_raw": "ItemsC", "sideU": "SideU", "sideB": "SideB",
    "sideC": "SideC", "lexIdf_vs_ctr": "LexIdfCtr", "lexBm25_vs_ctr": "LexBmCtr",
    "lexIdf_vs_zca": "LexIdfZca", "lexBm25_vs_zca": "LexBmZca", "zca_vs_pop": "ZcaPop",
    "hybZca_vs_pop": "HybPop", "hybZca_vs_hybRaw": "HybRaw", "knn_vs_zca": "KnnZca",
    "knn_vs_pop": "KnnPop", "knnHyb_vs_knn": "KnnHybKnn", "knnHyb_vs_pop": "KnnHybPop",
}  # fmt: skip
LO_ABBR = {
    "idf_overlap_title": "Idf", "bm25_title": "Bm", "strict_idf_h2_window": "Strict",
}  # fmt: skip


def bootstrap_share(d: str, num: str, den: str = "zca_train_users_a1.0", n_boot: int = 2000):
    """Point estimate and 95% percentile interval (users resampled) of the share, in percent, of
    the gain of ``den`` over raw that ``num`` recovers, from the per-user file of query_norm."""
    df = pd.read_csv(
        GRAN / f"query_norm_{d}_gte.csv.gz", usecols=["user_id", "gt", "condition", "nDCG@10"]
    )
    w = df[df["gt"] == "exact"].pivot(index="user_id", columns="condition", values="nDCG@10")
    x, y = (w[num] - w["raw"]).to_numpy(), (w[den] - w["raw"]).to_numpy()
    rng = np.random.default_rng(SEED)
    ratios = []
    for _ in range(n_boot // 250):
        idx = rng.integers(0, len(x), (250, len(x)))
        ratios.append(x[idx].mean(axis=1) / y[idx].mean(axis=1))
    lo, hi = np.percentile(np.concatenate(ratios), [2.5, 97.5])
    return 100 * x.mean() / y.mean(), 100 * lo, 100 * hi


def round3(a: A) -> None:
    D, R = DOM, a.R
    t = lambda x: " / ".join(str(v) for v in x)  # noqa: E731
    qsrc = GL + "stats_query_norm.csv"
    # ───────────── side x source matrix ─────────────
    for L, d in D.items():
        raw = a.Q(d, "raw")
        zu = a.Q(d, "zca_train_users_a1.0")
        for src, tag in (("U", "U"), ("B", "B"), ("C", "C")):
            for side in ("items", "query"):
                cond = pt.MATRIX_KEY[(side, src)]
                r = pt.matrix_test(R, d, f"matrix:{side}_{src}_vs_raw")
                S = side.capitalize()
                N(
                    f"Mx{S}{tag}{L}",
                    f(a.Q(d, cond), 4),
                    GL + "summary_query_norm.csv",
                    f"{d} {cond} ndcg10",
                )
                N(
                    f"Mx{S}{tag}Rel{L}",
                    rel(a.Q(d, cond) / raw - 1),
                    GL + "summary_query_norm.csv",
                    f"{cond} / raw - 1",
                )
                N(
                    f"Mx{S}{tag}P{L}",
                    pv(r["p_holm_signflip"]),
                    qsrc,
                    f"family matrix, matrix:{side}_{src}_vs_raw: p_holm_signflip",
                )
                N(f"Mx{S}{tag}Nnz{L}", th(r["n_nonzero_diff"]), qsrc, "same: n_nonzero_diff")
        N(f"MxRawK{L}", f(1e3 * raw, 2), GL + "summary_query_norm.csv", f"{d} raw ndcg10 x1e3")
        N(f"MxItemsUK{L}", f(1e3 * a.Q(d, "zca_catalog_train_users"), 2), GL + "summary_query_norm.csv", f"{d} zca_catalog_train_users ndcg10 x1e3")  # fmt: skip
        N(f"MxItemsUShare{L}", f(100 * (a.Q(d, "zca_catalog_train_users") - raw) / (zu - raw), 0),
          GL + "summary_query_norm.csv", f"{d}: (zca_catalog_train_users - raw) / (zca_train_users_a1.0 - raw) in percent")  # fmt: skip
        for src in "UBC":
            r = pt.matrix_test(R, d, f"matrix:side_{src}")
            k = f"matrix:side_{src}"
            N(f"MxSide{src}Diff{L}", u3(r["mean_diff"]), qsrc, f"{d} {k}: mean_diff")
            N(f"MxSide{src}Ci{L}", u3ci(r["ci95_low"], r["ci95_high"]), qsrc, f"{d} {k}: ci95")
            N(f"MxSide{src}P{L}", pv(r["p_holm_signflip"]), qsrc, f"{d} {k}: p_holm_signflip")
            N(f"MxSide{src}Nnz{L}", th(r["n_nonzero_diff"]), qsrc, f"{d} {k}: n_nonzero_diff")
        r = pt.matrix_test(R, d, "matrix:source_items_U_vs_C")
        N(
            f"MxSrcItemsDiff{L}",
            u3(r["mean_diff"]),
            qsrc,
            f"{d} matrix:source_items_U_vs_C: mean_diff",
        )
        N(f"MxSrcItemsP{L}", pv(r["p_holm_signflip"]), qsrc, f"{d} matrix:source_items_U_vs_C: p")
    for key in ("items_U_vs_raw", "items_B_vs_raw", "side_U", "side_B", "side_C"):
        tb = a.tally(f"matrix:{key}", "matrix")
        N("MxTally" + "".join(w.capitalize() for w in key.split("_")), t(tb), GL + "tally_query_norm.csv",
          f"matrix:{key}, music, verified: better / ns / worse")  # fmt: skip
    # bootstrap shares
    for L, d in D.items():
        for tag, num in (("Center", "center_train_users"), ("Books", "zca_books_catalog_a1.0")):
            p_, lo, hi = bootstrap_share(d, num)
            N(f"{tag}ShareCi{L}", f"{lo:.0f}\\% to {hi:.0f}\\%", "results/granularity/query_norm_<d>_gte.csv.gz",
              f"{d}: 95% percentile bootstrap (2000 resamples of users, seed {SEED}) of the share ({p_:.1f}% point)")  # fmt: skip
    # R@100 of the hybrid in movies (reported in the text, not only in the appendix table)
    for tag, ref in (("Pop", "popularity"), ("Raw", "hybrid_raw")):
        r = pt._stat_row(R.hyb_stats, domain="movies", encoder=GTE, metric="r100",
                         condition="hybrid_zca_profile", reference=ref)  # fmt: skip
        N(f"HybRHun{tag}DiffMovies", u3(r["mean_diff"]), GL + "stats_hybrid.csv", f"movies r100 hybrid_zca_profile vs {ref}: mean_diff")  # fmt: skip
        N(
            f"HybRHun{tag}PMovies",
            pv(r["p_holm_signflip"]),
            GL + "stats_hybrid.csv",
            "p_holm_signflip",
        )
    # ───────────── validation replication ─────────────
    rep, tot, tsig, flag = {}, {}, {}, {}
    for L, d in D.items():
        rep[L], tot[L], tsig[L], flag[L] = 0, 0, 0, 0
        for fam, test, _ in pt.VAL_ROWS:
            tr = pt._stat_row(
                R.qn_stats, domain=d, encoder=GTE, text=TC, metric="ndcg10", family=fam, test=test
            )
            vr = pt._stat_row(
                R.qn_val, domain=d, encoder=GTE, text=TC, metric="ndcg10", family=fam, test=test
            )
            rep[L] += int(pt.replicates(tr, vr))
            tsig[L] += int(pt.mark(tr["p_holm_signflip"], tr["mean_diff"]) != "")
            flag[L] += int(pt.ci_holm_disagree(vr))
            tot[L] += 1
        N(f"ValRep{L}", str(rep[L]), GL + "stats_query_norm_val.csv", f"{d}: contrasts of Table val-replication significant in the same direction on TEST and VAL")  # fmt: skip
        N(f"ValTestSig{L}", str(tsig[L]), GL + "stats_query_norm.csv", f"{d}: contrasts of Table val-replication significant on TEST")  # fmt: skip
        N(f"ValCiHolm{L}", str(flag[L]), GL + "stats_query_norm_val.csv", f"{d}: val rows where the CI excludes 0 and Holm p >= 0.05, or the reverse")  # fmt: skip
    N(
        "ValRepTot",
        str(tot["Music"]),
        GL + "stats_query_norm_val.csv",
        "contrasts per domain in the replication table",
    )
    for L, d in D.items():
        N(
            f"ValUsers{L}",
            th(a.run[d]["n_eval_users"]["val"]),
            GL + f"{d}_gte_titlecat.json",
            "n_eval_users.val",
        )
    # ───────────── splits ─────────────
    ms = R.ms
    seeds = sorted(pd.read_csv(GRAN / "multisplit_per_split.csv")["seed"].unique())
    N("MsExtra", str(len(seeds) - 1), GL + "multisplit_per_split.csv", "number of splits other than the main one")  # fmt: skip
    N(
        "MsSeedList",
        f"{seeds[0]} to {seeds[-1]}",
        GL + "multisplit_per_split.csv",
        "min and max seed",
    )
    N("MsNSplits", str(int(ms["n_splits"].max())), GL + "multisplit_summary.csv", "n_splits")
    for cid, name in MS_NAME.items():
        for L, d in D.items():
            r = pt._stat_row(ms, contrast=cid, domain=d)
            if r is None:
                continue
            src = GL + "multisplit_summary.csv"
            N(
                f"Ms{name}{L}Mean",
                sg3(r["mean_diff_mean"]),
                src,
                f"{cid}, {d}: mean_diff_mean (x1e3)",
            )
            N(
                f"Ms{name}{L}Sd",
                f(1e3 * r["mean_diff_sd"], 2),
                src,
                f"{cid}, {d}: mean_diff_sd (x1e3)",
            )
            N(f"Ms{name}{L}Min", sg3(r["mean_diff_min"]), src, f"{cid}, {d}: mean_diff_min (x1e3)")
            N(f"Ms{name}{L}Max", sg3(r["mean_diff_max"]), src, f"{cid}, {d}: mean_diff_max (x1e3)")
            N(f"Ms{name}{L}Better", str(int(r["n_better"])), src, f"{cid}, {d}: n_better")
            N(f"Ms{name}{L}Worse", str(int(r["n_worse"])), src, f"{cid}, {d}: n_worse")
            N(f"Ms{name}{L}Ns", str(int(r["n_ns"])), src, f"{cid}, {d}: n_ns")
    sh = pd.read_csv(GRAN / "multisplit_centring_share.csv").set_index("domain")
    for L, d in D.items():
        for k, name in (("mean", "Mean"), ("sd", "Sd"), ("min", "Min"), ("max", "Max")):
            N(
                f"MsCtrShare{L}{name}",
                f(100 * sh.loc[d, f"share_{k}"], 0),
                GL + "multisplit_centring_share.csv",
                f"{d}: {k} over splits of qCtr_vs_raw / qU_vs_raw (percent)",
            )
    lv = pd.read_csv(GRAN / "multisplit_levels.csv")
    for L, d in D.items():
        for cond, tag in (("raw", "Raw"), ("zca_train_users_a1.0", "Zca"), ("popularity", "Pop")):
            x = lv[(lv["domain"] == d) & (lv["condition"] == cond)]["ndcg10"]
            N(
                f"MsLv{tag}{L}Min",
                f(1e3 * x.min(), 2),
                GL + "multisplit_levels.csv",
                f"{d} {cond}: min over splits (x1e3)",
            )
            N(
                f"MsLv{tag}{L}Max",
                f(1e3 * x.max(), 2),
                GL + "multisplit_levels.csv",
                f"{d} {cond}: max over splits (x1e3)",
            )
        h = lv[(lv["domain"] == d) & (lv["condition"] == "zca_train_users_a1.0")]["n_hit10"]
        n = lv[(lv["domain"] == d) & (lv["condition"] == "zca_train_users_a1.0")]["n_users"]
        N(
            f"MsHitPctMin{L}",
            f(100 * (h / n).min(), 1),
            GL + "multisplit_levels.csv",
            f"{d} zca hit users / users: min",
        )
        N(
            f"MsHitPctMax{L}",
            f(100 * (h / n).max(), 1),
            GL + "multisplit_levels.csv",
            f"{d} zca hit users / users: max",
        )
    # ───────────── lexical overlap and semantic residual ─────────────
    lo = a.R.lo
    shares, jacs, exps = [], [], []
    for L, d in D.items():
        A_ = lo[d]["A_hit_overlap"]
        N(f"LoZcaHits{L}", str(A_["zca_train_users_a1.0"]["idf_overlap_title"]["n_a"]), GL + f"lexical_overlap_{d}.json", "A_hit_overlap.zca_train_users_a1.0.*.n_a")  # fmt: skip
        for lex, ab in LO_ABBR.items():
            r = A_["zca_train_users_a1.0"][lex]
            src = GL + f"lexical_overlap_{d}.json"
            N(f"Lo{ab}Hits{L}", str(r["n_b"]), src, f"A_hit_overlap.zca_train_users_a1.0.{lex}.n_b")
            N(f"Lo{ab}Both{L}", str(r["both"]), src, "... .both")
            N(f"Lo{ab}Exp{L}", f(r["expected_stratified"], 1), src, "... .expected_stratified")
            N(f"Lo{ab}Jac{L}", f(r["jaccard"], 2), src, "... .jaccard")
            N(f"Lo{ab}Share{L}", f(100 * r["share_a_in_b"], 0), src, "... .share_a_in_b (percent)")
            shares.append(100 * r["share_a_in_b"])
            jacs.append(r["jaccard"])
            exps.append(100 * r["expected_stratified"] / r["n_a"])
    N(
        "LoShareMin",
        f(min(shares), 0),
        GL + "lexical_overlap_*.json",
        "min over domains and scorers of share_a_in_b (ZCA)",
    )
    N(
        "LoShareMax",
        f(max(shares), 0),
        GL + "lexical_overlap_*.json",
        "max over domains and scorers",
    )
    N("LoJacMin", f(min(jacs), 2), GL + "lexical_overlap_*.json", "min jaccard (ZCA)")
    N("LoJacMax", f(max(jacs), 2), GL + "lexical_overlap_*.json", "max jaccard (ZCA)")
    N("LoChanceMax", f(max(exps), 0), GL + "lexical_overlap_*.json", "max over domains and scorers of expected_stratified / n_a (percent)")  # fmt: skip
    for L, d in D.items():
        B_ = lo[d]["B_reachability"]["0.01"]
        src = GL + f"lexical_overlap_{d}.json"
        for cond, tag in (("raw", "Raw"), ("center_train_users", "Ctr"), ("zca_train_users_a1.0", "Zca"),
                          ("idf_overlap_title", "Idf"), ("bm25_title", "Bm"), ("popularity", "Pop")):  # fmt: skip
            r = B_["rankings"][cond]
            N(f"LoReach{tag}{L}", f(100 * r["hit_reachable_share"], 0), src, f"B_reachability.0.01.rankings.{cond}.hit_reachable_share (percent)")  # fmt: skip
            N(f"LoPairs{tag}{L}", str(r["hit_pairs"]), src, f"... .{cond}.hit_pairs")
        N(f"LoReachRel{L}", f(100 * B_["relevant_reachable_share"], 1), src, "B_reachability.0.01.relevant_reachable_share (percent)")  # fmt: skip
        N(f"LoReachCat{L}", f(100 * B_["catalog_reachable_share"], 1), src, "B_reachability.0.01.catalog_reachable_share (percent)")  # fmt: skip
        N(
            f"LoResidUsers{L}",
            th(B_["users_with_residual"]),
            src,
            "B_reachability.0.01.users_with_residual",
        )
        N(
            f"LoReachUsers{L}",
            th(B_["users_with_reachable"]),
            src,
            "B_reachability.0.01.users_with_reachable",
        )
        for cond, tag in (("raw", "Raw"), ("center_train_users", "Ctr"), ("zca_train_users_a1.0", "Zca"),
                          ("idf_overlap_title", "Idf"), ("bm25_title", "Bm"), ("popularity", "Pop")):  # fmt: skip
            for gt, g in (("resid", "Resid"), ("reach", "Reach")):
                r = pt._stat_row(R.lo_sum, domain=d, gt=gt, condition=cond)
                N(
                    f"Lo{g}Nd{tag}{L}",
                    f(r["nDCG@10"], 4),
                    GL + "summary_lexical_overlap.csv",
                    f"{d} {gt} {cond} nDCG@10",
                )
                N(f"Lo{g}Hits{tag}{L}", str(int(r["n_hit10"])), GL + "summary_lexical_overlap.csv", f"{d} {gt} {cond} n_hit10")  # fmt: skip
        for test, tag in (("zca_vs_raw", "ZcaRaw"), ("zca_vs_center", "ZcaCtr"), ("center_vs_raw", "CtrRaw"),
                          ("zca_vs_idf_overlap_title", "ZcaIdf"), ("zca_vs_bm25_title", "ZcaBm")):  # fmt: skip
            r = pt._stat_row(R.lo_stats, domain=d, gt="resid", metric="ndcg10", test=test)
            N(
                f"LoResid{tag}Diff{L}",
                u3(r["mean_diff"]),
                GL + "stats_lexical_overlap.csv",
                f"{d} resid {test}: mean_diff",
            )
            N(
                f"LoResid{tag}Ci{L}",
                u3ci(r["ci95_low"], r["ci95_high"]),
                GL + "stats_lexical_overlap.csv",
                "ci95",
            )
            N(
                f"LoResid{tag}P{L}",
                pv(r["p_holm_signflip"]),
                GL + "stats_lexical_overlap.csv",
                "p_holm_signflip",
            )
            N(
                f"LoResid{tag}Nnz{L}",
                th(r["n_nonzero_diff"]),
                GL + "stats_lexical_overlap.csv",
                "n_nonzero_diff",
            )
            r = pt._stat_row(R.lo_stats, domain=d, gt="reach", metric="ndcg10", test=test)
            N(
                f"LoReach{tag}Diff{L}",
                u3(r["mean_diff"]),
                GL + "stats_lexical_overlap.csv",
                f"{d} reach {test}: mean_diff",
            )
            N(
                f"LoReach{tag}P{L}",
                pv(r["p_holm_signflip"]),
                GL + "stats_lexical_overlap.csv",
                "p_holm_signflip",
            )
        r = pt._stat_row(R.lo_stats, domain=d, gt="resid@0.001", metric="ndcg10", test="zca_vs_raw")
        sens_src = GL + "stats_lexical_overlap.csv"
        N(
            f"LoSensZcaRawDiff{L}",
            u3(r["mean_diff"]),
            sens_src,
            f"{d} resid@0.001 zca_vs_raw: mean_diff",
        )
        N(f"LoSensZcaRawCi{L}", u3ci(r["ci95_low"], r["ci95_high"]), sens_src, "ci95")
        N(f"LoSensZcaRawP{L}", pv(r["p_holm_signflip"]), sens_src, "p_holm_signflip")
        # masked residual (reachable items removed before ranking), three rarity thresholds
        gsrc = GL + "stats_lexical_overlap.csv"
        for gt, ft in (("resid@0.001", "Tenth"), ("resid", "One"), ("resid@0.05", "Five")):
            for test, tag in (("masked_zca_vs_raw", "ZcaRaw"), ("masked_center_vs_raw", "CtrRaw"),
                              ("masked_zca_vs_center", "ZcaCtr")):  # fmt: skip
                r = pt._stat_row(R.lo_stats, domain=d, gt=gt, metric="ndcg10", test=test)
                k = f"{d} {gt} {test}"
                N(f"LoMask{tag}{ft}Diff{L}", u3(r["mean_diff"]), gsrc, f"{k}: mean_diff")
                N(f"LoMask{tag}{ft}Ci{L}", u3ci(r["ci95_low"], r["ci95_high"]), gsrc, f"{k}: ci95")
                N(f"LoMask{tag}{ft}P{L}", pv(r["p_holm_signflip"]), gsrc, f"{k}: p_holm_signflip")
                N(f"LoMask{tag}{ft}Mde{L}", u3(r["mde80"]).replace("+", ""), gsrc, f"{k}: mde80")
                N(f"LoMask{tag}{ft}Nnz{L}", th(r["n_nonzero_diff"]), gsrc, f"{k}: n_nonzero_diff")
            r = pt._stat_row(R.lo_stats, domain=d, gt=gt, metric="ndcg10", test="zca_vs_raw")
            k = f"{d} {gt} zca_vs_raw (unmasked)"
            N(f"LoSens{ft}ZcaRawDiff{L}", u3(r["mean_diff"]), gsrc, f"{k}: mean_diff")
            N(f"LoSens{ft}ZcaRawP{L}", pv(r["p_holm_signflip"]), gsrc, f"{k}: p_holm_signflip")
        r = pt._stat_row(R.lo_stats, domain=d, gt="resid", metric="ndcg10", test="zca_vs_raw")
        N(f"LoResidZcaRawMde{L}", u3(r["mde80"]).replace("+", ""), gsrc, f"{d} resid zca_vs_raw: mde80")  # fmt: skip
        # overlap comparator: the share of the raw / centred hit users that also hit lexically
        for arm, atag in (("raw", "Raw"), ("center_train_users", "Ctr")):
            for lex, ab in LO_ABBR.items():
                r = lo[d]["A_hit_overlap"][arm][lex]
                N(f"Lo{ab}Share{atag}{L}", f(100 * r["share_a_in_b"], 0),
                  GL + f"lexical_overlap_{d}.json", f"A_hit_overlap.{arm}.{lex}.share_a_in_b (percent)")  # fmt: skip
        # share of the ZCA gain over raw (exact nDCG) that lies on the residual set, by hit pairs
        N(f"LoPairsGain{L}", str(B_["rankings"]["zca_train_users_a1.0"]["hit_pairs"] - B_["rankings"]["raw"]["hit_pairs"]),
          src, "hit pairs of ZCA minus hit pairs of raw")  # fmt: skip
        extra_reach = (B_["rankings"]["zca_train_users_a1.0"]["hit_pairs_reachable"]
                       - B_["rankings"]["raw"]["hit_pairs_reachable"])  # fmt: skip
        N(
            f"LoPairsGainReach{L}",
            str(extra_reach),
            src,
            "reachable hit pairs of ZCA minus those of raw",
        )
    zr = [100 * lo[d]["B_reachability"]["0.01"]["rankings"]["zca_train_users_a1.0"]["hit_reachable_share"] for d in D.values()]  # fmt: skip
    N("LoReachZcaMin", f(min(zr), 0), GL + "lexical_overlap_*.json", "min over domains of the ZCA hit_reachable_share (percent)")  # fmt: skip
    N("LoReachZcaMax", f(max(zr), 0), GL + "lexical_overlap_*.json", "max over domains of the ZCA hit_reachable_share (percent)")  # fmt: skip
    sens = lo["music"]["B_reachability"]
    N(
        "LoRareFrac",
        f(100 * lo["music"]["rare_fraction"], 0),
        GL + "lexical_overlap_music.json",
        "rare_fraction (percent)",
    )
    N("LoRareFracSens", f(100 * lo["music"]["rare_sensitivity"][0], 1), GL + "lexical_overlap_music.json", "rare_sensitivity[0] (percent)")  # fmt: skip
    del sens

    # ───────────── closed-form ridge bridge ─────────────
    for L, d in D.items():
        src = GL + "stats_bridge.csv"
        for c, tag in (("bridge_ridge", "Br"), ("raw", "Raw"), ("zca_train_users_a1.0", "Zca")):
            N(f"Br{tag}Nd{L}", f(float(R.br.loc[(d, TC, c)]["ndcg10"]), 4), GL + "summary_bridge.csv", f"{d} {c} ndcg10")  # fmt: skip
        for ref, tag in (("raw", "Raw"), ("zca_train_users_a1.0", "Zca")):
            r = pt._stat_row(R.br_stats, domain=d, metric="ndcg10", test=f"bridge_ridge_vs_{ref}")
            N(f"Br{tag}Diff{L}", u3(r["mean_diff"]), src, f"{d} bridge_ridge_vs_{ref}: mean_diff")
            N(f"Br{tag}Ci{L}", u3ci(r["ci95_low"], r["ci95_high"]), src, "ci95")
            N(f"Br{tag}P{L}", pv(r["p_holm_signflip"]), src, "p_holm_signflip")
        sel = json.loads((GRAN / f"bridge_{d}_gte.json").read_text())["selected_scale"]
        N(f"BrScale{L}", f"{sel:g}", GL + f"bridge_{d}_gte.json", "selected_scale (lambda / mean eigenvalue)")  # fmt: skip

    # nonlinear (MLP) bridge of the books pairs, against raw, the ZCA query and the ridge bridge
    for L, d in D.items():
        src = GL + "stats_bridge.csv"
        N(f"BrMlpNd{L}", f(float(R.br.loc[(d, TC, "bridge_mlp")]["ndcg10"]), 4), GL + "summary_bridge.csv", f"{d} bridge_mlp ndcg10")  # fmt: skip
        for ref, tag in (("raw", "Raw"), ("zca_train_users_a1.0", "Zca"), ("bridge_ridge", "Ridge")):
            r = pt._stat_row(R.br_stats, domain=d, metric="ndcg10", test=f"bridge_mlp_vs_{ref}")
            N(f"BrMlp{tag}Diff{L}", u3(r["mean_diff"]), src, f"{d} bridge_mlp_vs_{ref}: mean_diff")
            N(f"BrMlp{tag}Ci{L}", u3ci(r["ci95_low"], r["ci95_high"]), src, "ci95")
            N(f"BrMlp{tag}P{L}", pv(r["p_holm_signflip"]), src, "p_holm_signflip")
        sel = json.loads((GRAN / f"bridge_{d}_gte.json").read_text())["mlp_selected"]
        N(f"BrMlpSel{L}", sel.removeprefix("bridge_mlp_").replace("_", ", "), GL + f"bridge_{d}_gte.json", "mlp_selected (width, weight decay)")  # fmt: skip
    for L, d in D.items():
        bridge_cl_numbers(R, d, L, "")
        z = "zca_train_users_a1.0"
        for ref, tag in (("raw", "Raw"), (z, "Zca")):  # popularity list against the content queries
            r = pt._flip(pt._test(R.knn_stats, d, f"popularity_vs_{ref}"))
            N(f"PopC{tag}Diff{L}", u3(r["mean_diff"]), GL + "stats_knn.csv", f"{d} {ref} minus popularity: mean_diff")  # fmt: skip
            N(f"PopC{tag}Ci{L}", u3ci(r["ci95_low"], r["ci95_high"]), GL + "stats_knn.csv", "ci95")
            N(f"PopC{tag}P{L}", pv(r["p_holm_signflip"]), GL + "stats_knn.csv", "p_holm_signflip")
    N("CiHolmLegend", r"$^{\circ}$ the 95\% CI and the Holm test disagree", "paper_tables.ci_holm_disagree", "legend of the circle marker")  # fmt: skip
    source2_numbers(a)


def bridge_cl_numbers(Rx, d: str, L: str, pre: str) -> None:
    """Contrastive bridge (plain and query-centred after the bridge): TEST nDCG@10, the paired
    tests against raw, the ZCA query, the ridge and MLP bridges, the VAL selection and the sanity
    check, for the pair ``d`` (``pre="STwo"`` for the movies-as-source files)."""
    sfx = "_source2" if pre else ""
    src, z = GL + f"stats_bridge{sfx}.csv", "zca_train_users_a1.0"
    for c, tag in (("bridge_cl", "Cl"), ("bridge_cl_centred", "ClC")):
        N(f"{pre}Br{tag}Nd{L}", f(float(Rx.br.loc[(d, TC, c)]["ndcg10"]), 4), GL + f"summary_bridge{sfx}.csv", f"{d} {c} ndcg10")  # fmt: skip
    for b, ref, tag in (("cl", "raw", "ClRaw"), ("cl", z, "ClZca"), ("cl", "bridge_ridge", "ClRidge"),
                        ("cl", "bridge_mlp", "ClMlp"), ("cl_centred", "raw", "ClCRaw"),
                        ("cl_centred", z, "ClCZca"), ("cl_centred", "bridge_ridge", "ClCRidge"),
                        ("cl_centred", "bridge_cl", "ClCCl")):  # fmt: skip
        r = pt._stat_row(Rx.br_stats, domain=d, metric="ndcg10", test=f"bridge_{b}_vs_{ref}")
        N(f"{pre}Br{tag}Diff{L}", u3(r["mean_diff"]), src, f"{d} bridge_{b}_vs_{ref}: mean_diff")
        N(f"{pre}Br{tag}Ci{L}", u3ci(r["ci95_low"], r["ci95_high"]), src, "ci95")
        N(f"{pre}Br{tag}P{L}", pv(r["p_holm_signflip"]), src, "p_holm_signflip")
        N(f"{pre}Br{tag}CiHolm{L}", str(int(pt.ci_holm_disagree(r))), src, "CI excludes 0 xor Holm p < 0.05")  # fmt: skip
    j = json.loads((GRAN / f"bridgecl_{d}_gte.json").read_text())
    js = GL + f"bridgecl_{d}_gte.json"
    w, t, wd = j["selected"]
    N(f"{pre}BrClSel{L}", f"{w}, {t:g}, {wd:g}", js, "selected (width, temperature, weight decay)")
    s_ = j["sanity"]
    for key, tag in (("bridge_cl", "Cl"), ("raw", "Raw"), (z, "Zca"), ("random", "Rand")):
        N(f"{pre}BrSan{tag}{L}", f(1e3 * s_[key], 2), js, f"sanity.{key} (nDCG@10 x1000, held-out train item)")  # fmt: skip
    N(f"{pre}BrSanN{L}", th(s_["n_users"]), js, "sanity.n_users")
    N(f"{pre}BrSanRatio{L}", f(s_["bridge_cl"] / s_["random"], 0), js, "sanity: bridge_cl / random")


def source2_numbers(a: A) -> None:
    """Movies as SOURCE (movies2music, movies2games): sizes, baselines, the side x source matrix,
    kNN, lexical and the bridges, from the ``*_source2`` result files."""
    R2 = pt.Results("_source2")
    for L, d in {"MovMus": "movies2music", "MovGam": "movies2games"}.items():
        qn = lambda c, d=d: R2.qn.loc[(d, GTE, TC, c)]  # noqa: E731
        qs, qsrc = R2.qn_stats, GL + "stats_query_norm_source2.csv"
        N(f"STwoUsers{L}", th(int(qn("raw")["n_users"])), GL + "summary_query_norm_source2.csv", f"{d} test users")  # fmt: skip
        for c, tag in (("raw", "Raw"), ("popularity", "Pop"), ("zca_train_users_a1.0", "ZcaQ")):
            N(f"STwo{tag}Nd{L}", f(float(qn(c)["ndcg10"]), 4), GL + "summary_query_norm_source2.csv", f"{d} {c} ndcg10")  # fmt: skip
        kk = json.loads((GRAN / f"knn_{d}_gte.json").read_text())["selected_k"]
        N(f"STwoKnnNd{L}", f(float(R2.knn.loc[(d, TC, f"knn_raw_k{kk['raw']}")]["ndcg10"]), 4), GL + "summary_knn_source2.csv", f"{d} knn_raw_k{kk['raw']} ndcg10")  # fmt: skip
        N(f"STwoKnnK{L}", str(kk["raw"]), GL + f"knn_{d}_gte.json", "selected_k.raw")
        N(f"STwoBmNd{L}", f(float(R2.lex.loc[(d, TC, "bm25_title")]["ndcg10"]), 4), GL + "summary_lexical_source2.csv", f"{d} bm25_title ndcg10")  # fmt: skip
        for key, tag in (("query_U", "QU"), ("query_B", "QM"), ("query_C", "QC"),
                         ("items_U", "IU"), ("items_B", "IM"), ("items_C", "IC")):  # fmt: skip
            r = pt.matrix_test(R2, d, f"matrix:{key}_vs_raw")
            N(f"STwo{tag}Diff{L}", u3(r["mean_diff"]), qsrc, f"{d} matrix:{key}_vs_raw: mean_diff")
            N(f"STwo{tag}P{L}", pv(r["p_holm_signflip"]), qsrc, "p_holm_signflip")
        for src, tag in (("U", "U"), ("B", "M"), ("C", "C")):
            r = pt.matrix_test(R2, d, f"matrix:side_{src}")
            N(f"STwoSide{tag}Diff{L}", u3(r["mean_diff"]), qsrc, f"{d} matrix:side_{src}: mean_diff")
            N(f"STwoSide{tag}P{L}", pv(r["p_holm_signflip"]), qsrc, "p_holm_signflip")
        r = pt._stat_row(qs, domain=d, metric="ndcg10", test="post_hoc:zca_target_catalog_vs_raw")
        N(f"STwoTargetQDiff{L}", u3(r["mean_diff"]), qsrc, f"{d} post_hoc:zca_target_catalog_vs_raw: mean_diff")  # fmt: skip
        N(f"STwoTargetQP{L}", pv(r["p_holm_signflip"]), qsrc, "p_holm_signflip")
        kn_c = f"knn_center_train_users_k{kk['center_train_users']}"
        N(f"STwoKnnCNd{L}", f(float(R2.knn.loc[(d, TC, kn_c)]["ndcg10"]), 4), GL + "summary_knn_source2.csv", f"{d} {kn_c} ndcg10")  # fmt: skip
        N(f"STwoIdfNd{L}", f(float(R2.lex.loc[(d, TC, "idf_overlap_title")]["ndcg10"]), 4), GL + "summary_lexical_source2.csv", f"{d} idf_overlap_title ndcg10")  # fmt: skip
        N(f"STwoBothNd{L}", f(float(qn("zca_both_separate")["ndcg10"]), 4), GL + "summary_query_norm_source2.csv", f"{d} zca_both_separate ndcg10")  # fmt: skip
        zz = "zca_train_users_a1.0"
        for tag, df, fsrc, test in (
            ("PopZca", R2.knn_stats, "stats_knn_source2.csv", f"popularity_vs_{zz}"),
            ("KnnZca", R2.knn_stats, "stats_knn_source2.csv", f"knn_raw_k{kk['raw']}_vs_{zz}"),
            ("KnnCZca", R2.knn_stats, "stats_knn_source2.csv", f"{kn_c}_vs_{zz}"),
            ("BmZca", R2.lex_stats, "stats_lexical_source2.csv", f"bm25_title_vs_{zz}"),
            ("IdfZca", R2.lex_stats, "stats_lexical_source2.csv", f"idf_overlap_title_vs_{zz}"),
            ("BothQ", R2.qn_stats, "stats_query_norm_source2.csv", "c_both_sides_vs_query_only"),
            ("BothRaw", R2.qn_stats, "stats_query_norm_source2.csv", "vs_raw:zca_both_separate"),
        ):  # fmt: skip
            r = pt._test(df, d, test)
            N(f"STwo{tag}Diff{L}", u3(r["mean_diff"]), GL + fsrc, f"{d} {test}: mean_diff")
            N(f"STwo{tag}Ci{L}", u3ci(r["ci95_low"], r["ci95_high"]), GL + fsrc, "ci95")
            N(f"STwo{tag}P{L}", pv(r["p_holm_signflip"]), GL + fsrc, "p_holm_signflip")
        bridge_cl_numbers(R2, d, L, "STwo")
        bsrc = GL + "stats_bridge_source2.csv"
        for c, tag in (("bridge_ridge", "Ridge"), ("bridge_mlp", "Mlp")):
            N(f"STwoBr{tag}Nd{L}", f(float(R2.br.loc[(d, TC, c)]["ndcg10"]), 4), GL + "summary_bridge_source2.csv", f"{d} {c} ndcg10")  # fmt: skip
        for b, ref, tag in (("mlp", "raw", "MlpRaw"), ("mlp", "zca_train_users_a1.0", "MlpZca"),
                            ("mlp", "bridge_ridge", "MlpRidge"), ("ridge", "raw", "RidgeRaw")):  # fmt: skip
            r = pt._stat_row(R2.br_stats, domain=d, metric="ndcg10", test=f"bridge_{b}_vs_{ref}")
            N(f"STwoBr{tag}Diff{L}", u3(r["mean_diff"]), bsrc, f"{d} bridge_{b}_vs_{ref}: mean_diff")
            N(f"STwoBr{tag}P{L}", pv(r["p_holm_signflip"]), bsrc, "p_holm_signflip")


def tex_value(v: str) -> str:
    return v


def section_map(main_tex: str) -> dict[str, list[str]]:
    """Macro -> the sections (heading titles) of ``main.tex`` that use it."""
    used: dict[str, list[str]] = {}
    sec = "front matter"
    for line in main_tex.splitlines():
        m = re.match(r"\\(?:section|subsection)\{([^}]*)\}", line)
        if m:
            sec = m.group(1)
        elif line.startswith("\\begin{abstract}"):
            sec = "Abstract"
        for name in re.findall(r"\\num([A-Za-z]+)", line):
            if sec not in used.setdefault(name, []):
                used[name].append(sec)
    return used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--paper", required=True, type=Path, help="directory with main.tex")
    args = ap.parse_args()
    nums = build()
    main_tex = (args.paper / "main.tex").read_text()
    used = section_map(main_tex)
    missing = sorted(set(used) - set(nums))
    if missing:
        raise SystemExit(f"macros used in main.tex but not defined: {missing}")
    unused = sorted(set(nums) - set(used))
    body = "% Written by experiments/figures/paper_numbers.py from the result files. Do not edit.\n"
    for name in sorted(nums):
        body += f"\\newcommand{{\\num{name}}}{{{nums[name][0]}}}\n"
    (args.paper / "numbers.tex").write_text(body)
    md = [
        "# NUMBERS.md\n",
        "Every number printed in the prose of `main.tex` is a macro `\\num<Name>` defined in `numbers.tex`, "
        "which `experiments/figures/paper_numbers.py` writes from the result files of the code repository "
        "(paths relative to `libs/ACM - Asymmetric Catalog Whitening/`; `gte/titlecat` is the main run, "
        "`summary_*`, `stats_*` and `tally_*` files are in `results/granularity/`, mechanism files in "
        "`results/mechanism/`). The script fails if `main.tex` uses a macro it does not define; "
        "`--paper <dir>` regenerates this file. A relative gain is the mean of the condition over the mean of raw, minus 1; "
        "differences are mean paired differences (units stated in the text); p-values are Holm-adjusted p of the paired "
        "sign-flip test (the primary test) unless `wilcoxon` is named. Tables and figures are written by "
        "`experiments/figures/paper_tables.py` and `granularity_figures.py` from the same files and are not repeated here. "
        "Numbers of the literature audit (Appendix A) come from `review/tois/lit/*.md`, not from results.\n",
        "| Macro | As printed | Used in | Source file | Key / computation |",
        "|---|---|---|---|---|",
    ]
    for name in sorted(nums, key=lambda n: (used.get(n, ["~"])[0], n)):
        if name not in used:
            continue
        v, src, key = nums[name]
        md.append(
            f"| `\\num{name}` | `{v}` | {'; '.join(used[name])} | `{src}` | {key} |".replace(
                "|", "\\|", 0
            )
        )
    md.append(
        "\nNot listed: section, table, figure and equation numbers, years and citation keys; definitional constants "
        "(top-10 and top-100 cutoffs, depth-2 categories, 95% intervals, the 0.05 level, the exponent values 0 and 1, "
        "the embedding dimension, the number of encoders `10` in the Spearman test, `64` re-encoded rows in Appendix B, "
        "`10,000` sign flips, `0.99999`); the counts in the literature audit; and numbers that are thresholds "
        "(`10^{-3}`, `6\\times10^{-4}`) stated in table captions.\n"
    )
    (args.paper / "NUMBERS.md").write_text("\n".join(md))
    print(f"wrote numbers.tex ({len(nums)} macros, {len(used)} used) and NUMBERS.md")
    if unused:
        print("defined but not used:", ", ".join(unused))


if __name__ == "__main__":
    main()
