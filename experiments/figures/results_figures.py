"""
Renders the three main result figures of the paper as individual panels.

Panels:
  fig_a_music_results.pdf/png  — book→music nDCG@10 by method (horizontal bars)
  fig_b_multidomain.pdf/png    — multi-domain replication: catalog ZCA vs.
                                 best learned method
  fig_c_effect_sizes.pdf/png   — key effect sizes (rank-biserial r)

Supports: the headline results, multi-domain replication and effect-size
figures. The plotted values are hard-coded from the corresponding result
files (no inputs are read).

Output: results/figures/ (relative to the current directory; run from the
repository root).

Usage (from the repository root):
    python -m experiments.figures.results_figures
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

OUT_DIR = "results/figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── palette ───────────────────────────────────────────────────────────────────
C_BASE = "#2166ac"  # blue — ZCA catalog (proposed method)
C_LEARN = "#4dac26"  # green — best learned
C_PROF = "#d7191c"  # red — profile-side (degrades)
C_NEUT = "#888888"  # mid grey — other methods
C_RAND = "#cccccc"  # light grey — random
C_TEXT = "#1a1a1a"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.grid.axis": "x",
        "grid.alpha": 0.3,
        "grid.linewidth": 0.6,
    }
)


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE A — Book→Music results
# ═════════════════════════════════════════════════════════════════════════════
def fig_a():
    methods = [
        # (label,                       value,  color,   bold)
        ("Random", 0.1111, C_RAND, False),
        ("H2, raw mean pooling", 0.2691, C_NEUT, False),
        ("H2 + ZCA on user profile", 0.1941, C_PROF, False),
        ("H2 + ZCA on item catalog", 0.3476, C_BASE, True),
        ("H3, LLM synopsis — raw cat.", 0.0788, C_NEUT, False),
        ("H3 — whitened catalog", 0.1506, C_NEUT, False),
        ("H4, LLM entities — raw cat.", 0.0399, C_NEUT, False),
        ("H4 — whitened catalog", 0.1177, C_NEUT, False),
        ("H3 + best learned (MoE-PW)", 0.2117, C_LEARN, False),
        ("H4 + best learned (MoE-PW)", 0.1826, C_LEARN, False),
    ]
    methods_rev = list(reversed(methods))

    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    fig.subplots_adjust(left=0.32, right=0.88, top=0.88, bottom=0.10)

    labels = [m[0] for m in methods_rev]
    values = [m[1] for m in methods_rev]
    colors = [m[2] for m in methods_rev]
    bolds = [m[3] for m in methods_rev]
    y_pos = np.arange(len(labels))

    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.6, height=0.62)

    # reference line at the best result
    ax.axvline(0.3476, color=C_BASE, linewidth=1.2, linestyle="--", alpha=0.55, zorder=0)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("nDCG@10", fontsize=10)
    ax.set_title(
        "Book→Music: nDCG@10 by method\n(n = 2,731 users; † parameter-free contribution)",
        fontsize=10,
        fontweight="bold",
        pad=8,
    )
    ax.set_xlim(0, 0.44)
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # values to the right of each bar, without overlap
    for bar, val, bold, _col in zip(bars, values, bolds, colors):
        x_label = bar.get_width() + 0.006
        ax.text(
            x_label,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}" + (" †" if bold else ""),
            va="center",
            ha="left",
            fontsize=8.5,
            fontweight="bold" if bold else "normal",
            color=C_BASE if bold else C_TEXT,
        )

    # legend
    legend_items = [
        mpatches.Patch(color=C_BASE, label="ZCA catalog — ours (†)"),
        mpatches.Patch(color=C_LEARN, label="Best learned alignment"),
        mpatches.Patch(color=C_PROF, label="ZCA profile (degrades)"),
        mpatches.Patch(color=C_NEUT, label="Other methods"),
        mpatches.Patch(color=C_RAND, label="Random baseline"),
    ]
    ax.legend(
        handles=legend_items,
        fontsize=5.5,
        loc="lower right",
        framealpha=0.9,
        handlelength=0.7,
        handleheight=0.6,
        handletextpad=0.4,
        labelspacing=0.25,
        borderpad=0.5,
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(OUT_DIR, f"fig_a_music_results.{ext}"), bbox_inches="tight", dpi=220
        )
    plt.close(fig)
    print("fig_a saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE B — Multi-domain replication
# ═════════════════════════════════════════════════════════════════════════════
def fig_b():
    domains = ["Music\n(n=2,731)", "Movies\n(n=1,238)", "Games*\n(n=112/95)"]
    baselines = [0.3476, 0.2799, 0.4377]
    best_learned = [0.2117, 0.2816, 0.5687]
    sig_text = ["sig. worse", "sig. worse\n(14/14 pairs)", "sig. better\n(H3+PW)"]
    better = [False, False, True]

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    fig.subplots_adjust(left=0.10, right=0.97, top=0.85, bottom=0.22)

    x = np.arange(len(domains))
    w = 0.30

    bar_base = ax.bar(
        x - w / 2,
        baselines,
        w,
        label="ZCA catalog baseline",
        color=C_BASE,
        edgecolor="white",
        linewidth=0.6,
    )
    bar_learn = ax.bar(
        x + w / 2,
        best_learned,
        w,
        label="Best learned method",
        color=C_LEARN,
        edgecolor="white",
        linewidth=0.6,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(domains, fontsize=9.5)
    ax.set_ylabel("nDCG@10", fontsize=10)
    ax.set_ylim(0, 0.72)
    ax.set_title(
        "Multi-domain replication: ZCA catalog vs. best learned\n"
        "(Wilcoxon–Pratt + permutation, Holm-Bonferroni, k=14 pairs/domain)",
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.legend(fontsize=9, loc="upper left", framealpha=0.9, handlelength=1.1)

    # values inside the bars (at the top, inside)
    for bar in list(bar_base) + list(bar_learn):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h - 0.018,
            f"{h:.3f}",
            ha="center",
            va="top",
            fontsize=8.0,
            color="white",
            fontweight="bold",
        )

    # significance annotations above each pair of bars
    bracket_top_gap = 0.035
    for i, (bl, lv, sig, b) in enumerate(zip(baselines, best_learned, sig_text, better)):
        top = max(bl, lv) + bracket_top_gap
        color = C_LEARN if b else C_PROF
        # horizontal line of the bracket
        ax.plot([i - w / 2, i + w / 2], [top, top], color=color, linewidth=1.2)
        ax.plot([i - w / 2, i - w / 2], [top - 0.008, top], color=color, linewidth=1.2)
        ax.plot([i + w / 2, i + w / 2], [top - 0.008, top], color=color, linewidth=1.2)
        ax.text(
            i,
            top + 0.008,
            sig,
            ha="center",
            va="bottom",
            fontsize=7.5,
            color=color,
            fontweight="bold",
        )

    # footnote
    ax.text(
        0.5,
        -0.20,
        "* Games: nDCG@10 inflated by small catalog (15,600 items; random baseline = 0.50).\n"
        "  Significance tests paired by user intersection; raw cross-population means are not comparable.",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=7,
        color="#555555",
        style="italic",
    )

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT_DIR, f"fig_b_multidomain.{ext}"), bbox_inches="tight", dpi=220)
    plt.close(fig)
    print("fig_b saved")


# ═════════════════════════════════════════════════════════════════════════════
# FIGURE C — Key effect sizes (rank-biserial r)
# ═════════════════════════════════════════════════════════════════════════════
def fig_c():
    # (short label, r, color, p_label)
    effects = [
        ("H3: catalog whit. vs. raw catalog\n(music, n=2,731)", 0.576, C_BASE, "p<10⁻¹⁰³"),
        ("H4: catalog whit. vs. raw catalog\n(music, n=1,794)", 0.740, C_BASE, "p<10⁻¹¹⁰"),
        ("H3: MoE-PW vs. H2 catalog whit.\n(music, catalog wins)", 0.541, C_BASE, "p<10⁻²⁰"),
        ("H4: MoE-PW vs. H2 catalog whit.\n(music, catalog wins)", 0.565, C_BASE, "p<10⁻²⁰"),
        ("Games H×ZB interaction — PW\n(n=68 common users)", 0.630, C_LEARN, "p=6.3×10⁻⁶"),
        ("Games H×ZB interaction — MoE\n(n=68 common users)", 0.585, C_LEARN, "p=2.8×10⁻⁵"),
    ]
    effects_rev = list(reversed(effects))

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    fig.subplots_adjust(left=0.38, right=0.72, top=0.85, bottom=0.12)

    labels = [e[0] for e in effects_rev]
    values = [e[1] for e in effects_rev]
    colors = [e[2] for e in effects_rev]
    plabels = [e[3] for e in effects_rev]

    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, values, color=colors, edgecolor="white", linewidth=0.6, height=0.55)

    # Cohen thresholds — vertical line + label at its top, inside the plot
    thresholds = [(0.1, "small"), (0.3, "medium"), (0.5, "large")]
    for xv, lbl in thresholds:
        ax.axvline(xv, color="#bbbbbb", linewidth=0.8, linestyle=":", zorder=0)
        # y position = top of the last bar + small margin
        ax.annotate(
            lbl, xy=(xv, len(labels) - 0.5), fontsize=7, color="#888888", ha="center", va="bottom"
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlabel("Rank-biserial r  (effect size)", fontsize=10)
    ax.set_xlim(0, 1.05)
    ax.set_title(
        "Key effect sizes — Wilcoxon–Pratt + permutation test\n"
        "(all pairs: p < 0.001; Holm-Bonferroni corrected)",
        fontsize=9.5,
        fontweight="bold",
        pad=8,
    )
    ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))

    # values + p-value to the right
    for bar, val, plbl in zip(bars, values, plabels):
        ax.text(
            bar.get_width() + 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"r = {val:.3f}  ({plbl})",
            va="center",
            ha="left",
            fontsize=8.0,
            color=C_TEXT,
        )

    # legend placed to the right of the axes
    legend_items = [
        mpatches.Patch(
            color=C_BASE, label="Catalog whitening advantage\n(catalog beats learned / raw)"
        ),
        mpatches.Patch(
            color=C_LEARN, label="Games H×ZB interaction\n(ZB direction differs H3 vs H4)"
        ),
    ]
    ax.legend(
        handles=legend_items,
        fontsize=9,
        loc="center left",
        bbox_to_anchor=(1.03, 0.5),
        framealpha=0.95,
        handlelength=1.2,
        borderpad=1.0,
    )

    for ext in ("pdf", "png"):
        fig.savefig(
            os.path.join(OUT_DIR, f"fig_c_effect_sizes.{ext}"), bbox_inches="tight", dpi=220
        )
    plt.close(fig)
    print("fig_c saved")


# ═════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    fig_a()
    fig_b()
    fig_c()
    print(f"\nAll figures saved to: {OUT_DIR}/")
