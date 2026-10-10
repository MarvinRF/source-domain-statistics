"""
experiments/encoder_sweep/full_pipeline_round2.py
=================================================
Promotes four additional backbones from the lightweight screening
(data/processed/backbone_screening/screening_results[_round2].json) to the
full pipeline, raising the encoder sweep from n=5 to n=9 backbones so the
Spearman correlation between spectral anisotropy (erank / top-20% variance)
and the nDCG gain from catalog whitening has real statistical power.

Reuses `run_backbone` from experiments/encoder_sweep/full_pipeline.py (same
protocol as the first sweep: music domain, canonical_val_users.json n=2,731,
MAX_BOOKS_PROFILE=10, full-rank ZCA); the embedding/whitening/evaluation
logic is not duplicated.

Text prefixes: only e5-large-v2 uses one ("passage: "), kept for consistency
with the screening run (experiments/encoder_sweep/screening.py /
screening_round2.py).

Supports: the encoder sweep relating embedding anisotropy to the
catalog-whitening gain.

Output: data/processed/backbone_screening/full_pipeline_results_round2.json
        (resumable: backbones already in the file are skipped).

Usage:
    python -m experiments.encoder_sweep.full_pipeline_round2
    # or one backbone per process (recommended: CUDA memory not released
    # between sequential models in the same process can cause OOM, e.g. when
    # loading bge-m3 after e5-large-v2):
    python -m experiments.encoder_sweep.full_pipeline_round2 --only bge-m3
"""

import argparse
import json

from experiments.encoder_sweep.full_pipeline import OUT, log, run_backbone

OUT_JSON = OUT / "full_pipeline_results_round2.json"

BACKBONES = [
    ("e5-large-v2", "intfloat/e5-large-v2", "passage: ", 64),
    ("bge-m3", "BAAI/bge-m3", "", 16),  # batch_size reduced: OOM at 64 on a
    # 6GB GPU even in a fresh process
    # (~5GB resident for the model alone,
    # little headroom left).
    ("snowflake-arctic-embed-l", "Snowflake/snowflake-arctic-embed-l", "", 64),
    # jina-embeddings-v3: DROPPED -- 3 distinct failures (missing einops, corrupted
    # remote-code cache, then a genuine transformers==5.8.0/custom-code API
    # incompatibility -- 'XLMRobertaLoRA' object has no attribute
    # 'all_tied_weights_keys'). Fixing the last one needs a transformers downgrade,
    # too risky for the whole pinned pipeline.
    # Replaced with UAE-Large-V1 (BERT-large-based, no trust_remote_code needed,
    # smoke-tested OK) to still close the sweep at n=9.
    ("uae-large-v1", "WhereIsAI/UAE-Large-V1", "", 64),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--only",
        type=str,
        default=None,
        choices=[b[0] for b in BACKBONES],
        help="Run a single backbone in this process (recommended -- "
        "fresh CUDA context per backbone, avoids OOM from "
        "unreleased memory across sequential model loads).",
    )
    args = p.parse_args()

    results = {}
    if OUT_JSON.exists():
        results = json.load(open(OUT_JSON))
        log(f"Loaded existing partial results: {list(results.keys())}")

    backbones = [b for b in BACKBONES if b[0] == args.only] if args.only else BACKBONES

    for name, model_id, prefix, batch_size in backbones:
        if name in results:
            log(f"{name}: already processed, skipping.")
            continue
        results[name] = run_backbone(model_id, name, prefix=prefix, batch_size=batch_size)
        json.dump(results, open(OUT_JSON, "w"), indent=2)
        log(f"Checkpoint saved ({name}) to {OUT_JSON}")

    log(f"\nDONE. Final results in {OUT_JSON}")


if __name__ == "__main__":
    main()
