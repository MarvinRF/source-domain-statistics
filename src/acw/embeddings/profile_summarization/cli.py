"""Command-line entry point for LLM-summarized user profiles (H3/H4 variants).

An LLM summarizes each user's source-domain history into a natural-language
profile, which is then encoded with the embedding model.

Outputs (all in data/processed/), where {tag} is "" unless --run-tag is set,
in which case every filename gets a "__{run_tag}" suffix so a different
model/run can never overwrite a previous run's cache:
  user_profiles_{heuristic}{tag}.json          — raw LLM summaries per user
  user_profiles_{heuristic}{tag}.npy           — float32 embedding matrix (N_USERS, D)
  user_profile_index_{heuristic}{tag}.json     — {"user_id": row_index, ...}
  user_profiles_{heuristic}{tag}_metadata.json — per-run experiment metadata
  run_manifest.jsonl                           — append-only log of every run
                                                  (model, run_tag, timestamp,
                                                  user counts), regardless of
                                                  whether --run-tag was used
  debug_traces/{heuristic}/                    — per-user prompt + output traces

Usage:
  python -m acw.embeddings.profile_summarization.cli --heuristic h3
  python -m acw.embeddings.profile_summarization.cli --heuristic h4
  python -m acw.embeddings.profile_summarization.cli --heuristic h3_concrete --dry-run --limit 5
  python -m acw.embeddings.profile_summarization.cli --heuristic h4_concrete --model qwen2.5:14b
  python -m acw.embeddings.profile_summarization.cli --heuristic both
  # Generating with a different/larger model? Always tag the run so it
  # can't silently merge into an existing cache file:
  python -m acw.embeddings.profile_summarization.cli --heuristic h3 \\
      --model /path/to/llama-3.3-70b-awq --text-only --run-tag llama70b
"""

from __future__ import annotations

import argparse
import asyncio

from .config import DEFAULT_LLM_MODEL, EMBEDDING_MODEL
from .pipeline import async_main


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate H3/H4/H3_concrete/H4_concrete user profiles.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--heuristic",
        choices=[
            "h3",
            "h4",
            "h3_concrete",
            "h4_concrete",
            "h3_movies",
            "h4_movies",
            "h3_games",
            "h4_games",
            "both",
        ],
        default="both",
        help="Which heuristic(s) to generate.",
    )
    parser.add_argument("--model", default=DEFAULT_LLM_MODEL)
    parser.add_argument("--embedding-model", default=EMBEDDING_MODEL)
    parser.add_argument(
        "--device", default="cpu", help="Device for SentenceTransformer (cpu or cuda)"
    )
    parser.add_argument(
        "--all-users", action="store_true", help="Generate profiles for ALL overlapping users"
    )
    parser.add_argument(
        "--user-list",
        default=None,
        help=(
            "Path to a JSON file with an explicit list of user_ids (e.g. "
            "data/processed/canonical_val_users.json, the canonical "
            "validation population, n=2731). Takes precedence over "
            "--all-users. Use it to generate profiles only for the "
            "evaluated population instead of all ~24,420 overlapping "
            "users (~9x slower)."
        ),
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Only generate LLM texts and skip embedding (to keep the LLM server and the embedding model off the GPU at the same time)",
    )
    parser.add_argument(
        "--embed-only",
        action="store_true",
        help="Only embed already-cached texts and skip LLM generation (run after --text-only with the LLM server stopped)",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--run-tag",
        default=None,
        help=(
            "Tags output filenames with this suffix "
            "(user_profiles_{heuristic}__{run_tag}.json) so a run with a "
            "different model never overwrites a previous run's cache. "
            "Also recorded in the append-only run manifest "
            "(data/processed/run_manifest.jsonl). Omit to use the "
            "untagged filenames (default)."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()
