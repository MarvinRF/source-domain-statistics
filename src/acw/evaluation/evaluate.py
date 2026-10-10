"""Canonical evaluation entry point; all reported results are produced here.

Protocol:
    - Dataset      : Amazon Reviews 2023 (Books → CDs & Vinyl / Movies & TV /
                     Video Games, 5-core)
    - Users        : the canonical validation split (see generate_canonical_split)
    - Eligibility  : users whose ground truth spans at most MAX_GENRES=6 subgenres
    - Profile      : last MAX_BOOKS=10 books of the user's history
    - Ground truth : target items expanded by subgenre (``--gt genre``) or the
                     exact consumed items (``--gt exact``)
    - Metrics      : nDCG@K, P@K for K ∈ {5, 10, 20}; R@K for K ∈ {10, 50, 100}
    - Baselines    : random ranking (SEED=42), popularity, top-genre

Usage:
    python -m acw.evaluation.evaluate --all
    python -m acw.evaluation.evaluate --domain movies --all
    python -m acw.evaluation.evaluate \\
        --profiles data/processed/alignment/user_profiles_h3_pw_zb.npy \\
        --label "H3+PW-ZB"

See ``acw.evaluation.cli`` for the full list of arguments.
"""

from .cli import main

if __name__ == "__main__":
    main()
