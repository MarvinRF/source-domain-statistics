"""Disjoint 70/10/20 train/val/test user splits, one per target domain.

Eligible users have a non-empty exact ground truth (restricted to the indexed
target catalog) and at least one book in their history that has an embedding.
Eligible users are sorted, shuffled with ``acw.settings.SEED`` and cut into
train (70%), val (10%) and test (20%). The three lists are asserted pairwise
disjoint before anything is written.

Output (``data/splits/``): ``{domain}_{train,val,test}.json`` (sorted user-id
lists) and ``{domain}_stats.json`` (counts). The main split uses ``--seed 42``
(the default); any other seed S writes the same files to ``data/splits/seed{S}/``.
The experiments read the split of ``ACW_SPLIT_SEED`` (``acw.settings.SPLIT_SEED``).

Usage::

    python -m acw.data.splits --domain music
    python -m acw.data.splits --domain movies --seed 43
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from acw.embeddings.generate_h2_profiles.io import load_user_history
from acw.evaluation.config import resolve_paths
from acw.evaluation.loader import build_exact_ground_truth
from acw.settings import MAX_BOOKS_PROFILE, ROOT, SEED, SPLIT_SEED

SPLITS_DIR = ROOT / "data" / "splits"
SPLIT_NAMES = ("train", "val", "test")
FRACTIONS = (0.7, 0.1, 0.2)
DOMAINS = ("music", "movies", "games", "movies2music", "movies2games")


def split_users(
    users: list[str], seed: int = SEED, fractions: tuple[float, float, float] = FRACTIONS
) -> dict[str, list[str]]:
    """Shuffle ``users`` (after sorting) and cut them into disjoint, sorted
    train/val/test lists; test takes the remainder after train and val."""
    if abs(sum(fractions) - 1.0) > 1e-9:
        raise ValueError(f"fractions must sum to 1, got {fractions}")
    ordered = sorted(set(users))
    perm = np.random.default_rng(seed).permutation(len(ordered))
    n_train = round(fractions[0] * len(ordered))
    n_val = round(fractions[1] * len(ordered))
    cuts = {
        "train": perm[:n_train],
        "val": perm[n_train : n_train + n_val],
        "test": perm[n_train + n_val :],
    }
    splits = {name: sorted(ordered[i] for i in idx) for name, idx in cuts.items()}
    assert_disjoint(splits)
    assert sum(len(v) for v in splits.values()) == len(ordered)
    return splits


def assert_disjoint(splits: dict[str, list[str]]) -> None:
    names = list(splits)
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            overlap = set(splits[a]) & set(splits[b])
            assert not overlap, f"splits {a} and {b} share {len(overlap)} users"


def splits_dir(seed: int | None = None) -> Path:
    """Directory of the splits of ``seed`` (default: ``ACW_SPLIT_SEED``): the main seed
    uses ``data/splits/`` itself, any other seed its subdirectory ``seed{S}/``."""
    seed = SPLIT_SEED if seed is None else seed
    return SPLITS_DIR if seed == SEED else SPLITS_DIR / f"seed{seed}"


def make_splits(domain: str, seed: int = SEED) -> dict:
    """Build, check and write the splits of ``domain`` for ``seed``; return the stats dict."""
    cfg = resolve_paths(domain)
    exact = build_exact_ground_truth(cfg)
    history = load_user_history(domain)
    with open(cfg.source_idx_path) as f:
        book_idx = json.load(f)

    has_book = {u for u, books in history.items() if any(a in book_idx for a in books)}
    has_window_book = {
        u for u, books in history.items() if any(a in book_idx for a in books[-MAX_BOOKS_PROFILE:])
    }
    eligible = sorted(set(exact) & has_book)
    splits = split_users(eligible, seed)

    stats = {
        "domain": domain,
        "seed": seed,
        "fractions": dict(zip(SPLIT_NAMES, FRACTIONS)),
        "n_history_users": len(history),
        "n_exact_gt_users": len(exact),
        "n_with_embedded_book": len(has_book),
        "n_eligible": len(eligible),
        "n_eligible_without_window_profile": len(set(eligible) - has_window_book),
        **{f"n_{name}": len(users) for name, users in splits.items()},
    }
    out = splits_dir(seed)
    out.mkdir(parents=True, exist_ok=True)
    for name, users in splits.items():
        (out / f"{domain}_{name}.json").write_text(json.dumps(users))
    (out / f"{domain}_stats.json").write_text(json.dumps(stats, indent=2))
    return stats


def split_path(domain: str, split: str, seed: int | None = None) -> Path:
    if split not in SPLIT_NAMES:
        raise ValueError(f"split must be one of {SPLIT_NAMES}, got {split!r}")
    return splits_dir(seed) / f"{domain}_{split}.json"


def load_split(domain: str, split: str, seed: int | None = None) -> list[str]:
    """Sorted user ids of ``split`` (train|val|test) for ``domain`` (seed: see ``splits_dir``)."""
    path = split_path(domain, split, seed)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run: python -m acw.data.splits --domain {domain} --seed "
            f"{SPLIT_SEED if seed is None else seed}"
        )
    return json.loads(path.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", choices=DOMAINS, required=True)
    ap.add_argument("--seed", type=int, default=SEED, help="shuffle seed (default: the main split)")
    args = ap.parse_args()
    print(json.dumps(make_splits(args.domain, args.seed), indent=2))


if __name__ == "__main__":
    main()
