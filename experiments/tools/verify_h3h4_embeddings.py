"""
Integrity check of an H3/H4 (music) user-profile embedding file against the
project's GTEEncoder (src/acw/embeddings/profile_summarization/gte_encoder.py),
the single GTE embedding implementation in the package.

Re-embeds a deterministic (fixed-seed) sample of users and reports the cosine
similarity between each fresh vector and the vector stored in the given .npy
file (mean/min/max and the number of users with cosine < 0.95).

Supports: reproducibility of the user-profile embeddings used by every H3/H4
experiment.

Inputs (under data/processed/): profile texts (.json), embeddings (.npy) and
the user index (.json). By default (no --npy-source/--index-source) it checks
the canonical user_profiles_{h}.npy / user_profile_index_{h}.json.
Output: printed to stdout only. Run it twice and compare the outputs to
confirm determinism before trusting the result.

Usage:
  python -m experiments.tools.verify_h3h4_embeddings \\
      --heuristic h3 --n 200 \\
      --text-source user_profiles_h3__llama70b_20260626_h4fix.json \\
      --npy-source user_profiles_h3_fixed.npy \\
      --index-source user_profile_index_h3_fixed.json
"""

import argparse
import json
import random

import numpy as np
import torch

from acw.settings import ROOT  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"

from acw.embeddings.profile_summarization.gte_encoder import GTEEncoder  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heuristic", choices=["h3", "h4"], required=True)
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--text-source",
        default=None,
        help="Profile-text .json file name (default: user_profiles_{h}.json)",
    )
    ap.add_argument(
        "--npy-source", default=None, help=".npy file to verify (default: user_profiles_{h}.npy)"
    )
    ap.add_argument(
        "--index-source",
        default=None,
        help="User-index file name (default: user_profile_index_{h}.json)",
    )
    args = ap.parse_args()

    index_path = PROCESSED_DIR / (args.index_source or f"user_profile_index_{args.heuristic}.json")
    json_path = PROCESSED_DIR / (args.text_source or f"user_profiles_{args.heuristic}.json")
    npy_path = PROCESSED_DIR / (args.npy_source or f"user_profiles_{args.heuristic}.npy")

    index = json.loads(index_path.read_text())  # user_id -> row index
    texts_by_user = json.loads(json_path.read_text())
    stored = np.load(npy_path)
    print(f"device={args.device} npy={npy_path.name} texts={json_path.name}", flush=True)

    valid_users = [u for u in index.keys() if texts_by_user.get(u) is not None]
    rng = random.Random(args.seed)
    sample = rng.sample(valid_users, min(args.n, len(valid_users)))

    encoder = GTEEncoder(device=args.device)
    sample_texts = [texts_by_user[u] for u in sample]
    new_embs = encoder.encode(sample_texts, batch_size=32, show_progress_bar=True)

    sims = []
    for j, uid in enumerate(sample):
        row = index[uid]
        stored_vec = stored[row]
        new_vec = new_embs[j]
        denom = np.linalg.norm(stored_vec) * np.linalg.norm(new_vec)
        cos = float(np.dot(stored_vec, new_vec) / denom) if denom > 0 else float("nan")
        sims.append(cos)

    sims_arr = np.array(sims)
    print(f"heuristic={args.heuristic} n_sampled={len(sample)}")
    print(f"mean_cos_sim={sims_arr.mean():.6f} min={sims_arr.min():.6f} max={sims_arr.max():.6f}")
    n_divergent = int((sims_arr < 0.95).sum())
    print(f"n_divergent(cos<0.95)={n_divergent} ({100 * n_divergent / len(sims_arr):.1f}%)")
    divergent_users = [sample[i] for i, s in enumerate(sims) if s < 0.95]
    print(f"divergent_user_ids_sample={divergent_users[:10]}")


if __name__ == "__main__":
    main()
