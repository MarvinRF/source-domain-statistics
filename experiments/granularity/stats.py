"""
Paired statistics on the per-user results of experiments.granularity.run_corrections.

For every run (domain, encoder, text) found in results/granularity/ and every
relevance granularity (``gt``: exact, d3, d2, d1), on the TEST split only and for
nDCG@10 and Recall@100:

  stats_vs_raw.csv         each condition vs ``raw``. Every comparison reports the
                           number of users, the means, ``n_nonzero_diff`` (users
                           whose two values differ), the mean difference with its
                           bootstrap 95% CI, the PRIMARY paired sign-flip test on
                           the mean (``p_signflip``), the secondary Wilcoxon-Pratt
                           test (``p_wilcoxon``) and the rank-biserial r (over the
                           non-zero differences only). Holm is applied separately
                           within each family = (domain, encoder, text, gt, metric)
                           over the corrections, EXCLUDING ``pca_catalog``, to the
                           sign-flip p (``p_holm_signflip``, primary) and to the
                           Wilcoxon p (``p_holm_wilcoxon``). ``p_holm`` is the
                           earlier Wilcoxon-Holm over the family WITH ``pca_catalog``,
                           kept for the tables that already use it. Baselines are
                           reported unadjusted. ``verified`` is false for the
                           encoders without a passed re-encoding audit.
  stats_vs_popularity.csv  the same, ``raw`` and each correction vs ``popularity``
                           (Holm over those conditions).
  stats_vs_top_category.csv
                           the same against the constant ``top_category`` list.
  stats_tally.csv          counts of corrections significantly better / worse / not
                           different (Holm sign-flip p < 0.05) against each of the
                           three references, per (gt, metric) and in total, separately
                           for verified and unverified encoders; ``pca_catalog`` and
                           the baselines are not counted.
  stats_gap.csv            granularity gap per correction: relative gain over
                           ``raw`` at depth d minus relative gain at ``exact``
                           (relative gain = (mean(cond) - mean(raw)) / mean(raw)),
                           with a paired bootstrap CI that resamples users
                           jointly across granularities and conditions.
  summary_test.csv         one row per run/granularity/condition: mean nDCG@10 and
                           R@100, difference to raw, CI, n_nonzero_diff, Holm p
                           (Wilcoxon and sign-flip), r.
  stats_provenance.json    provenance (code commit, package versions, sha256 of the
                           per-user CSVs analysed) of the five tables above.

All granularities of a run are evaluated on the same users, so the gap is a
paired quantity. Tests use acw.evaluation.significance (paired_row, add_holm_columns, holm,
relative_gain_gap); the bootstrap uses N_BOOT resamples and the sign-flip test
N_FLIP.

Usage:
    python -m experiments.granularity.stats
    python -m experiments.granularity.stats --runs music_gte_titlecat
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
import pandas as pd

from acw.evaluation.significance import add_holm_columns, holm, paired_row, relative_gain_gap
from acw.provenance import provenance
from acw.settings import GRANULARITY_DIR
from experiments.granularity.common import tally
from experiments.granularity.run_corrections import UNVERIFIED_ENCODERS

log = logging.getLogger("granularity_stats")

OUT_DIR = GRANULARITY_DIR
BASELINES = ("random", "popularity", "top_category")
METRICS = {"nDCG@10": "ndcg10", "R@100": "r100"}
GT_ORDER = ("exact", "d3", "d2", "d1")
N_BOOT = 2_000
N_FLIP = 10_000
EXCLUDED = (
    "pca_catalog",
)  # invalid control: not in the Holm families of the primary test, nor in tallies
DOMAINS = ("music", "movies", "games")


def run_stems() -> list[str]:
    """Stems of the grid per-user result files (not legacy, hybrid, kNN, ...): domain_encoder_text."""
    stems = [p.name[: -len(".csv.gz")] for p in OUT_DIR.glob("*.csv.gz")]
    return sorted(s for s in stems if s.count("_") == 2 and s.split("_")[0] in DOMAINS)


def _tests(wide: pd.DataFrame, ref: str, conditions: list[str]) -> list[dict]:
    """Paired tests of each condition against ``ref`` plus Holm over ``conditions``."""
    rows = []
    for c in conditions:
        rows.append(
            {"condition": c, **paired_row(wide[c].to_numpy(), wide[ref].to_numpy(), N_BOOT, N_FLIP)}
        )
    return rows


def _add_holm(rows: list[dict], family: set[str]) -> None:
    """``p_holm`` (Wilcoxon, whole family) and the Holm columns of both tests without EXCLUDED."""
    members = [r for r in rows if r["condition"] in family]
    p_adj, _ = holm([r["p_wilcoxon"] for r in members])
    for r in rows:
        r["p_holm"] = np.nan
    for r, p in zip(members, p_adj):
        r["p_holm"] = float(p)
    add_holm_columns(rows, [r for r in members if r["condition"] not in EXCLUDED])


def analyse(stem: str) -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    domain, encoder, text = stem.split("_")
    df = pd.read_csv(
        OUT_DIR / f"{stem}.csv.gz", usecols=["user_id", "split", "gt", "condition", *METRICS]
    )
    df = df[df["split"] == "test"]
    order = list(df["condition"].unique())
    corrections = [c for c in order if c != "raw" and c not in BASELINES]
    tag = {
        "domain": domain,
        "encoder": encoder,
        "text": text,
        "verified": encoder not in UNVERIFIED_ENCODERS,
    }
    vs_raw, vs_pop, vs_top, gap = [], [], [], []
    wides: dict[tuple[str, str], pd.DataFrame] = {}
    gts = [g for g in GT_ORDER if g in set(df["gt"])]
    for gt in gts:
        sub = df[df["gt"] == gt]
        for metric, short in METRICS.items():
            wide = sub.pivot(index="user_id", columns="condition", values=metric)
            wide = wide[order].sort_index()
            assert not wide.isna().any().any(), f"{stem} {gt}: users missing conditions"
            wides[(gt, metric)] = wide
            meta = {**tag, "gt": gt, "metric": short}
            r = _tests(wide, "raw", corrections + [b for b in BASELINES if b in order])
            _add_holm(r, set(corrections))
            vs_raw += [{**meta, **x} for x in r]
            r = _tests(wide, "popularity", ["raw", *corrections])
            _add_holm(r, {"raw", *corrections})
            vs_pop += [{**meta, **x} for x in r]
            r = _tests(wide, "top_category", ["raw", *corrections])
            _add_holm(r, {"raw", *corrections})
            vs_top += [{**meta, **x} for x in r]
    for metric, short in METRICS.items():
        a = wides[("exact", metric)]
        for gt in gts:
            if gt == "exact":
                continue
            b = wides[(gt, metric)]
            assert a.index.equals(b.index), f"{stem}: users differ between exact and {gt}"
            out = relative_gain_gap(
                a["raw"].to_numpy(),
                a[corrections].to_numpy(),
                b["raw"].to_numpy(),
                b[corrections].to_numpy(),
                n_resamples=N_BOOT,
            )
            for i, c in enumerate(corrections):
                gap.append(
                    {
                        **tag,
                        "metric": short,
                        "gt": gt,
                        "condition": c,
                        "n": len(a),
                        "gain_exact": out["gain_a"][i],
                        "gain_gt": out["gain_b"][i],
                        "gap": out["gap"][i],
                        "ci95_low": out["ci95_low"][i],
                        "ci95_high": out["ci95_high"][i],
                    }
                )
    return vs_raw, vs_pop, vs_top, gap


def summarise(vs_raw: pd.DataFrame) -> pd.DataFrame:
    """One row per run/granularity/condition with both metrics side by side."""
    keys = ["domain", "encoder", "text", "gt", "condition"]
    wide = vs_raw.pivot(index=keys, columns="metric")
    out = pd.DataFrame(index=wide.index)
    for short in METRICS.values():
        out[f"{short}"] = wide[("mean_cond", short)]
        out[f"{short}_delta"] = wide[("mean_diff", short)]
        out[f"{short}_ci_low"] = wide[("ci95_low", short)]
        out[f"{short}_ci_high"] = wide[("ci95_high", short)]
        out[f"{short}_n_nonzero"] = wide[("n_nonzero_diff", short)]
        out[f"{short}_p_holm"] = wide[("p_holm", short)]
        out[f"{short}_p_holm_signflip"] = wide[("p_holm_signflip", short)]
        out[f"{short}_r"] = wide[("rank_biserial_r", short)]
    out["n"] = wide[("n", "ndcg10")]
    return out.reset_index().rename(columns={"gt": "granularity"})


def tallies(vs_raw: pd.DataFrame, vs_pop: pd.DataFrame, vs_top: pd.DataFrame) -> pd.DataFrame:
    """Better / worse / not-different counts of the corrections against each reference."""
    skip = {*EXCLUDED, *BASELINES, "raw"}
    parts = []
    for ref, df in (("raw", vs_raw), ("popularity", vs_pop), ("top_category", vs_top)):
        rows = df[~df["condition"].isin(skip)].to_dict("records")
        by_level = tally(rows, ["gt", "metric"])
        total = tally(rows, [])
        parts.append(by_level.assign(reference=ref))
        parts.append(total.assign(reference=ref, gt="all", metric="all"))
    out = pd.concat(parts, ignore_index=True)
    first = ["reference", "gt", "metric", "encoders"]
    return out[first + [c for c in out.columns if c not in first]]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runs", nargs="*", help="run stems (domain_encoder_text); default: all")
    ap.add_argument("--tag", default="", help="suffix of the output files (e.g. _source2)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s", datefmt="%H:%M:%S")

    vs_raw, vs_pop, vs_top, gap = [], [], [], []
    stems = args.runs or run_stems()
    assert not args.tag or args.runs, "--tag needs --runs"
    for stem in stems:
        a, b, c, d = analyse(stem)
        vs_raw += a
        vs_pop += b
        vs_top += c
        gap += d
        log.info(f"{stem}: {len(a)} vs-raw tests")
    vs_raw_df, vs_pop_df, vs_top_df, gap_df = map(pd.DataFrame, (vs_raw, vs_pop, vs_top, gap))

    # Raw rows (no test against itself): add their means so the summary is complete.
    raw_rows = vs_raw_df[vs_raw_df["condition"] == "random"].copy()
    raw_rows["condition"] = "raw"
    raw_rows["mean_cond"] = raw_rows["mean_ref"]
    for col in (
        "mean_diff", "ci95_low", "ci95_high", "n_nonzero_diff", "p_signflip", "p_wilcoxon",
        "p_holm", "p_holm_signflip", "p_holm_wilcoxon", "rank_biserial_r",
    ):  # fmt: skip
        raw_rows[col] = np.nan
    summary = summarise(pd.concat([vs_raw_df, raw_rows], ignore_index=True))

    for name, frame in (
        ("stats_vs_raw", vs_raw_df),
        ("stats_vs_popularity", vs_pop_df),
        ("stats_vs_top_category", vs_top_df),
        ("stats_gap", gap_df),
        ("stats_tally", tallies(vs_raw_df, vs_pop_df, vs_top_df)),
        ("summary_test", summary),
    ):
        frame.to_csv(OUT_DIR / f"{name}{args.tag}.csv", index=False, float_format="%.6g")
        log.info(f"saved {OUT_DIR / (name + args.tag)}.csv ({len(frame)} rows)")
    prov = {"runs": stems, "provenance": provenance([OUT_DIR / f"{s}.csv.gz" for s in stems])}
    (OUT_DIR / f"stats_provenance{args.tag}.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
