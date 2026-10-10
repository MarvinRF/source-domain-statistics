"""Command-line entry point for H2 (mean-pooling) user profile generation.

Each user profile is the L2-normalized mean of the embeddings of the user's
most recent source-domain items (up to MAX_BOOKS_PROFILE).

Outputs (all in data/processed/):
  user_profiles_h2.npy               — float32 matrix (N_USERS, DIM)
  user_profile_index_h2.json         — {"user_id": row_index, ...}
  user_profiles_h2_metadata.json     — experiment metadata and reproducibility info

Usage:
  python -m acw.embeddings.generate_h2_profiles.cli
  python -m acw.embeddings.generate_h2_profiles.cli --all-users
"""

import argparse

from .pipeline import main


def parse_args():
    p = argparse.ArgumentParser(description="Generate H2 mean-pooling profiles")
    p.add_argument(
        "--all-users",
        action="store_true",
        help="Generate profiles for ALL overlapping users (not just the canonical 2000-user sample)",
    )
    p.add_argument(
        "--domain",
        default="music",
        choices=["music", "movies", "games"],
        help="Source domain for user history (default: music)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(all_users=args.all_users, domain=args.domain)
