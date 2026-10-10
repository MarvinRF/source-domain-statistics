"""
Computes spectral predictors (effective rank, top-20 variance fraction) of
the album-catalog embeddings for every encoder backbone in the sweep, and
their Spearman correlation with the nDCG@10 gain from catalog whitening.

Supports: the encoder-sweep analysis of whether embedding anisotropy
predicts the direction and size of the catalog-whitening gain (spectral
predictor table and its Spearman statistics).

The five original backbones (GTE-large, Qwen3-8B, BGE-large, all-MiniLM,
nomic-embed) are listed below with their measured gains. If
data/processed/backbone_screening/full_pipeline_results_round2.json exists
(produced by experiments/encoder_sweep/full_pipeline_round2.py), the four
additional backbones from that run (e5-large-v2, bge-m3,
snowflake-arctic-embed-l, uae-large-v1) are appended, taking n from 5 to 9.
With n this small the Spearman test has very limited power, so a
non-significant p-value is not evidence of no relationship. This script is
the single source of truth for the reported correlation.

Inputs: album-catalog embeddings for each backbone (paths in BACKBONE_FILES
        and ROUND2_EMB_TEMPLATE), relative to the repository root.
Output: results/statistical_tests/spectral_predictors.json (also prints a
        LaTeX-ready table).

Usage (from the repository root):
    python -m experiments.encoder_sweep.spectral_predictors
"""

import json
import os

import numpy as np
from scipy.stats import spearmanr

BACKBONE_FILES = {
    "GTE-large": "data/processed/album_embeddings.npy",
    "Qwen3-8B": "data/processed/album_embeddings_qwen3.npy",
    "BGE-large": "data/processed/album_embeddings_bge-large.npy",
    "all-MiniLM": "data/processed/backbone_screening/albums_embeddings_minilm.npy",
    "nomic-embed": "data/processed/backbone_screening/albums_embeddings_nomic.npy",
}

# Direction of catalog whitening for each encoder (from full_pipeline_results.json / paper)
WHITENING_DIRECTION = {
    "GTE-large": "+",
    "Qwen3-8B": "+",
    "BGE-large": "+",
    "all-MiniLM": "-",
    "nomic-embed": "-",
}

NDCG_DELTA = {
    "GTE-large": "+29.2%",
    "Qwen3-8B": "+59.6%",
    "BGE-large": "+7.9%",
    "all-MiniLM": "-10.1%",
    "nomic-embed": "-13.0%",
}

ROUND2_RESULTS_PATH = "data/processed/backbone_screening/full_pipeline_results_round2.json"
ROUND2_EMB_TEMPLATE = "data/processed/backbone_screening/albums_embeddings_{name}.npy"


def _load_round2_backbones():
    """
    Appends the 4 backbones already screened (screening_results[_round2].json)
    and promoted to a full pipeline run (experiments/encoder_sweep/full_pipeline_round2.py)
    to BACKBONE_FILES/WHITENING_DIRECTION/NDCG_DELTA in place, if the results file
    exists. No-op (script still runs at n=5) if the background run hasn't finished.
    """
    if not os.path.exists(ROUND2_RESULTS_PATH):
        print(
            f"[round2] {ROUND2_RESULTS_PATH} not found yet -- running at n={len(BACKBONE_FILES)}."
        )
        return
    round2 = json.load(open(ROUND2_RESULTS_PATH))
    for name, r in round2.items():
        emb_path = ROUND2_EMB_TEMPLATE.format(name=name)
        if not os.path.exists(emb_path):
            print(f"[round2] SKIP {name}: {emb_path} not found")
            continue
        delta_pct = r["delta"] / r["raw"] * 100
        BACKBONE_FILES[name] = emb_path
        WHITENING_DIRECTION[name] = "+" if delta_pct > 0 else "-"
        NDCG_DELTA[name] = f"{delta_pct:+.1f}%"
    print(f"[round2] Added {len(round2)} backbones -- running at n={len(BACKBONE_FILES)}.")


def compute_spectral(emb: np.ndarray) -> dict:
    """Effective rank (Roy & Vetterli 2007) and top-20 variance fraction."""
    emb = emb.astype(np.float32)
    centered = emb - emb.mean(axis=0, keepdims=True)
    N = emb.shape[0]
    cov = (centered.T @ centered) / (N - 1)
    eigvals = np.linalg.eigvalsh(cov)[::-1]  # descending
    pos = eigvals[eigvals > 1e-10]
    p = pos / pos.sum()
    erank = float(np.exp(-np.sum(p * np.log(p + 1e-12))))
    top20_frac = float(pos[:20].sum() / pos.sum() * 100)
    return {"erank": round(erank, 1), "top20_var_pct": round(top20_frac, 1)}


def _parse_delta_pct(ndcg_delta_str: str) -> float:
    return float(ndcg_delta_str.replace("%", "").replace("+", ""))


def main():
    _load_round2_backbones()

    results = {}
    for name, path in BACKBONE_FILES.items():
        if not os.path.exists(path):
            print(f"SKIP {name}: {path} not found")
            continue
        emb = np.load(path)
        sp = compute_spectral(emb)
        sp["whitening_effect"] = WHITENING_DIRECTION[name]
        sp["ndcg_delta"] = NDCG_DELTA[name]
        results[name] = sp
        print(
            f"{name}: erank={sp['erank']}, top20%={sp['top20_var_pct']}%, effect={sp['whitening_effect']}"
        )

    out_path = "results/statistical_tests/spectral_predictors.json"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved to {out_path}")

    # LaTeX table fragment
    print("\n% LaTeX table:")
    print(r"\begin{tabular}{@{}lrrcc@{}}")
    print(r"\toprule")
    print(r"Encoder & Eff.\ rank & Top-20 var.\ (\%) & Whitening effect \\ \midrule")
    for name, r in results.items():
        arrow = r"$\uparrow$" if r["whitening_effect"] == "+" else r"$\downarrow$"
        print(f"{name} & {r['erank']} & {r['top20_var_pct']} & {arrow} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")

    # ── Spearman correlation: erank / top-20% var vs. nDCG gain ────────────
    # A correlation test at n=5 has essentially no power (p>0.28 is
    # uninformative, not evidence of "no relationship"). This block is the
    # single source of truth for the reported numbers; re-running it once the
    # 4 extra backbones are available (n=9) requires no code change.
    names = list(results.keys())
    eranks = [results[n]["erank"] for n in names]
    top20s = [results[n]["top20_var_pct"] for n in names]
    deltas = [_parse_delta_pct(results[n]["ndcg_delta"]) for n in names]

    n = len(names)
    rho_erank, p_erank = spearmanr(eranks, deltas)
    rho_top20, p_top20 = spearmanr(top20s, deltas)

    spearman_summary = {
        "n": n,
        "backbones": names,
        "erank_vs_delta": {"rho": round(float(rho_erank), 3), "p": round(float(p_erank), 4)},
        "top20_vs_delta": {"rho": round(float(rho_top20), 3), "p": round(float(p_top20), 4)},
        "power_caveat": (
            f"n={n} gives a Spearman test very limited power; a non-significant "
            "p-value here reflects absence of power, not evidence of no "
            "relationship between spectral properties and whitening direction."
        ),
    }
    print(
        f"\nSpearman (n={n}): erank vs. delta rho={rho_erank:.3f} (p={p_erank:.4f}); "
        f"top-20% var vs. delta rho={rho_top20:.3f} (p={p_top20:.4f})"
    )
    print(f"CAVEAT: {spearman_summary['power_caveat']}")

    results["_spearman_summary"] = spearman_summary
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRe-saved (with Spearman summary) to {out_path}")


if __name__ == "__main__":
    main()
