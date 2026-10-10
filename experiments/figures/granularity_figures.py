"""
Figures of the paper, generated from the result files only.

  fig_granularity_ladder.pdf  relative nDCG@10 gain over ``raw`` along the ladder of
                              relevance granularities (exact -> d1), catalog-side ZCA
                              (alpha=1) vs query-side ZCA (statistics of the training users)
                              vs the constant top-category list, three domains,
                              GTE-large-en-v1.5.
  fig_query_norm.pdf          relative exact-level nDCG@10 gain over ``raw`` of ZCA^alpha
                              (alpha=0 is centring) as a function of alpha, for the query
                              normalised with the statistics of the training users, of the
                              book catalog or of the target catalog, and for the items
                              transformed with the target catalog's own statistics (curve)
                              or, at alpha = 1, with those of the users or of the books
                              (single points): the side x source matrix; three domains,
                              GTE-large-en-v1.5.

Reads results/granularity/{summary_test.csv, summary_query_norm.csv, stats_query_norm.csv,
<domain>_gte_titlecat.json}. A filled marker means Holm-adjusted sign-flip p < 0.05 against
``raw`` (the family of corrections of the grid, or the query-normalisation family of
stats_query_norm.csv); a hollow marker means not significant or not tested (arms that were
not fixed in advance have no test); the constant list is outside that family and is always
drawn hollow. Colour-blind-safe palette (Okabe-Ito); each
series also has its own marker and line style, so the figures read in greyscale. Vector PDF.

Usage (repository root):
    python -m experiments.figures.granularity_figures --out "<paper dir>"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from experiments.figures.paper_tables import (  # noqa: E402
    ALPHA,
    DNAME,
    DOMAINS,
    MATRIX_KEY,
    Results,
    levels_of,
    matrix_mark,
    qn_mark,
    qn_row,
    run_json,
)

BLUE, VERMILLION, ORANGE, GREY = "#0072B2", "#D55E00", "#E69F00", "#4d4d4d"
ALPHAS = (0.0, 0.25, 0.5, 0.75, 1.0)
ARM_NAMES = {  # source -> pattern of the arm name at exponent alpha
    "train_users": ("center_train_users", "zca_train_users_a{a}"),
    "books_catalog": ("center_books_catalog", "zca_books_catalog_a{a}"),
    "target_catalog": ("center_target_catalog", "zca_target_catalog_a{a}"),
    "items": ("center_catalog", "zca_catalog_a{a}"),
}
SERIES = (  # condition, legend label, colour, marker, line style
    ("zca_catalog_a1.0", "Catalog ZCA", BLUE, "o", "-"),
    ("zca_profile", "Query ZCA (users)", VERMILLION, "s", "--"),
    ("top_category", "Constant top-category list", GREY, "^", ":"),
)

plt.rcParams.update(
    {
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.5,
    }
)


def _draw(ax, xs, ys, filled, color, marker, ls) -> None:
    ax.plot(xs, ys, color=color, ls=ls, lw=1.2, zorder=2)
    for x, y, f in zip(xs, ys, filled):
        ax.plot(
            x, y, marker=marker, ms=5.5, color=color, mfc=color if f else "white", mew=1.1, zorder=3
        )


def _legend(fig, handles) -> None:
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        fontsize=7,
        handlelength=2.4,
        columnspacing=1.2,
        bbox_to_anchor=(0.5, 0.0),
    )


def fig_ladder(R: Results, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.7))
    for ax, d in zip(axes, DOMAINS):
        levels = levels_of(d)
        size = run_json(d)["relevance_size"]["test"]
        xs = list(range(len(levels)))
        for cond, _, color, marker, ls in SERIES:
            vals = [R.rel(d, "gte", "titlecat", g, cond) for g in levels]
            ys = [100 * v for v, _ in vals]
            filled = [cond != "top_category" and p < ALPHA for _, p in vals]
            _draw(ax, xs, ys, filled, color, marker, ls)
        ax.axhline(0, color="black", lw=0.7, zorder=1)
        ax.set_xticks(xs)
        ax.set_xticklabels(
            [f"{g}\n{size[g]['median_rel_pct_of_catalog']:.2g}%" for g in levels], fontsize=7
        )
        ax.set_xlim(-0.3, len(levels) - 0.7)
        ax.set_title(DNAME[d])
        if d == DOMAINS[0]:
            ax.set_ylabel("nDCG@10 gain over raw (%)")
    handles = [
        Line2D([], [], color=c, marker=m, ls=ls, mfc=c, ms=5.5, lw=1.2, label=lab)
        for _, lab, c, m, ls in SERIES
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            ls="",
            mfc="white",
            ms=5.5,
            label="hollow: not significant / untested",
        )
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.supxlabel("Relevance granularity (median relevant set, % of catalog)", fontsize=8, y=0.10)
    _legend(fig, handles)
    fig.savefig(
        out / "fig_granularity_ladder.pdf",
        metadata={"CreationDate": None, "Creator": None, "Producer": None},
    )
    plt.close(fig)


def _arm(source: str, a: float) -> str:
    center, zca = ARM_NAMES[source]
    return center if a == 0 else zca.format(a=f"{a:g}" if a < 1 else "1.0")


def fig_query_norm(R: Results, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.6), sharey=False)
    series = (  # source, label, colour, marker, line style
        ("train_users", "Query, users' statistics", VERMILLION, "s", "-"),
        ("books_catalog", "Query, books' statistics", ORANGE, "s", "-"),
        ("target_catalog", "Query, target statistics", BLUE, "s", "-"),
        ("items", "Items, target statistics", BLUE, "o", "--"),
    )
    points = (  # items at alpha = 1 with the statistics of the users / of the books
        ("U", "Items, users' statistics (alpha = 1)", VERMILLION, "o"),
        ("B", "Items, books' statistics (alpha = 1)", ORANGE, "o"),
    )
    for ax, d in zip(axes, DOMAINS):
        raw = float(qn_row(R, d, "raw")["ndcg10"])
        for source, _, color, marker, ls in series:
            ys, filled = [], []
            for a in ALPHAS:
                arm = _arm(source, a)
                ys.append(100 * (float(qn_row(R, d, arm)["ndcg10"]) / raw - 1))
                if a == 1.0:  # the alpha = 1 cells have their test in the matrix family
                    side, src = ("items" if source == "items" else "query"), {
                        "train_users": "U", "books_catalog": "B", "target_catalog": "C", "items": "C",
                    }[source]  # fmt: skip
                    filled.append(matrix_mark(R, d, side, src) != "")
                else:
                    filled.append(qn_mark(R, d, arm, source == "items") != "")
            _draw(ax, list(ALPHAS), ys, filled, color, marker, ls)
        for src, _, color, marker in points:
            y = 100 * (float(qn_row(R, d, MATRIX_KEY[("items", src)])["ndcg10"]) / raw - 1)
            ax.plot(1.0, y, marker=marker, ms=5.5, color=color, mew=1.1, ls="",
                    mfc=color if matrix_mark(R, d, "items", src) else "white", zorder=3)  # fmt: skip
        ax.axhline(0, color="black", lw=0.7, zorder=1)
        ax.set_xticks(list(ALPHAS))
        ax.set_xticklabels(["0", "", "0.5", "", "1"])
        ax.set_xlabel(r"exponent $\alpha$ (0 = centring)")
        ax.set_title(DNAME[d])
    axes[0].set_ylabel("nDCG@10 gain over raw (%)")
    handles = [
        Line2D([], [], color=c, marker=m, ls=ls, mfc=c, ms=5.5, lw=1.2, label=lab)
        for _, lab, c, m, ls in series
    ]
    handles += [
        Line2D([], [], color=c, marker=m, ls="", mfc=c, ms=5.5, label=lab)
        for _, lab, c, m in points
    ]
    handles.append(
        Line2D(
            [],
            [],
            color="black",
            marker="o",
            ls="",
            mfc="white",
            ms=5.5,
            label="hollow: not significant / not tested",
        )  # fmt: skip
    )
    fig.tight_layout(rect=(0, 0.26, 1, 1))
    fig.legend(
        handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=7,
        handlelength=2.4, columnspacing=1.2, bbox_to_anchor=(0.5, 0.0),
    )  # fmt: skip
    fig.savefig(
        out / "fig_query_norm.pdf",
        metadata={"CreationDate": None, "Creator": None, "Producer": None},
    )
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path, help="directory that receives the PDFs")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    R = Results()
    fig_ladder(R, args.out)
    fig_query_norm(R, args.out)
    print(f"wrote fig_granularity_ladder.pdf and fig_query_norm.pdf to {args.out}")


if __name__ == "__main__":
    main()
