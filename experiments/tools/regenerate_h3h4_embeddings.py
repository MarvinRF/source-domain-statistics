"""
Regenerate the canonical H3/H4 user-profile embeddings (music, and optionally
movies/games) with the project's GTEEncoder
(src/acw/embeddings/profile_summarization/gte_encoder.py).

GTEEncoder deterministically initialises the RoPE buffers of GTE-large; an
encoder that skips this step yields non-deterministic or corrupted
embeddings. This script contains no encoding logic of its own: it only reads
the profile texts, calls GTEEncoder (the single GTE embedding implementation
in the package) and writes the result.

Supports: the user-profile embeddings consumed by every H3/H4 experiment.

Input:  data/processed/user_profiles_{heuristic}__{run_tag}.json (profile texts)
Output: NEW files with a ``_fixed`` suffix, so the canonical files are never
        overwritten before they are checked with
        ``python -m experiments.tools.verify_h3h4_embeddings``:
          data/processed/user_profiles_{heuristic}_fixed_{run_tag}.npy
          data/processed/user_profile_index_{heuristic}_fixed_{run_tag}.json

Usage:
  python -m experiments.tools.regenerate_h3h4_embeddings --heuristic h3
  python -m experiments.tools.regenerate_h3h4_embeddings --heuristic h4
"""

import argparse
import json

import numpy as np
import torch

from acw.settings import ROOT  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"

from acw.embeddings.profile_summarization.gte_encoder import GTEEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--heuristic",
        choices=["h3", "h4", "h3_movies", "h4_movies", "h3_games", "h4_games"],
        required=True,
    )
    ap.add_argument("--run-tag", default="llama70b_20260626_h4fix")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    src_json = PROCESSED_DIR / f"user_profiles_{args.heuristic}__{args.run_tag}.json"
    texts_by_user = json.loads(src_json.read_text())
    ordered_users = [u for u, t in texts_by_user.items() if t is not None]
    texts = [texts_by_user[u] for u in ordered_users]
    print(
        f"heuristic={args.heuristic} users_total={len(texts_by_user)} users_valid={len(ordered_users)}",
        flush=True,
    )

    encoder = GTEEncoder(device=args.device)
    embeddings = encoder.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    assert not np.isnan(embeddings).any(), (
        "GTEEncoder produced NaN — aborting, not saving corrupted data."
    )

    out_npy = PROCESSED_DIR / f"user_profiles_{args.heuristic}_fixed_{args.run_tag}.npy"
    out_idx = PROCESSED_DIR / f"user_profile_index_{args.heuristic}_fixed_{args.run_tag}.json"
    np.save(out_npy, embeddings)
    out_idx.write_text(json.dumps({u: i for i, u in enumerate(ordered_users)}, ensure_ascii=False))
    print(f"saved {out_npy} shape={embeddings.shape}", flush=True)
    print(f"saved {out_idx}", flush=True)


if __name__ == "__main__":
    main()
