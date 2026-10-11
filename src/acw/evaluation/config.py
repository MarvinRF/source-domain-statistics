"""Evaluation protocol constants, per-domain path defaults and the registry of
known methods. ``DomainConfig`` and ``resolve_paths`` centralise all path
handling so that the CLI and pipeline never build paths themselves.

Usage::

    python -m acw.evaluation.evaluate --domain music --all
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Evaluation protocol (fixed; changing these changes every reported number)
# ─────────────────────────────────────────────────────────────────────────────

SEED: int = 42
N_TEST_USERS: int = 2000
MAX_BOOKS: int = 10
MAX_GENRES: int = 6
K_VALUES: list[int] = [5, 10, 20]
EPS: float = 1e-6

# ─────────────────────────────────────────────────────────────────────────────
# Root paths
# ─────────────────────────────────────────────────────────────────────────────

from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
RESULTS_DIR = ROOT / "results" / "evaluation"

# ─────────────────────────────────────────────────────────────────────────────
# DomainConfig
# ─────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DomainConfig:
    domain: str
    raw_dir: Path
    gt_path: Path
    items_meta_path: Path
    items_emb_path: Path
    items_idx_path: Path
    canonical_users_path: Path  # legacy fixed sample (n=2000), fallback only
    canonical_val_path: Path  # evaluation split (~20% of eligible users)
    canonical_train_path: Path  # training split (~80% of eligible users)
    category_depth: int
    # source catalog of the pair: the canonical books by default, the movies for ``movies2*``
    source_idx_path: Path = PROCESSED / "book_id_to_idx.json"
    source_meta_path: Path | None = None  # its titles (jsonl); None = the books file of the domain


# ─────────────────────────────────────────────────────────────────────────────
# Per-domain defaults
# ─────────────────────────────────────────────────────────────────────────────

_DOMAIN_DEFAULTS: dict[str, dict] = {
    "music": {
        "raw_dir": ROOT / "data" / "raw",
        "gt_filename": "ground_truth_real.json",
        "items_meta_filename": "albums_filtered.jsonl",
        "items_emb_filename": "album_embeddings.npy",
        "items_idx_filename": "album_id_to_idx.json",
        "canonical_users_filename": "canonical_test_users.json",  # legacy fallback
        "canonical_val_filename": "canonical_val_users.json",
        "canonical_train_filename": "canonical_train_users.json",
        "category_depth": 2,
    },
    "movies": {
        "raw_dir": ROOT / "data" / "raw" / "movies",
        "gt_filename": "ground_truth_real.json",
        "items_meta_filename": "movies_filtered.jsonl",
        "items_emb_filename": "movies/movies_embeddings_qwen.npy",
        "items_idx_filename": "movies/movies_id_to_idx_qwen.json",
        "canonical_users_filename": "canonical_test_users_movies.json",
        "canonical_val_filename": "canonical_val_users_movies.json",
        "canonical_train_filename": "canonical_train_users_movies.json",
        "category_depth": 2,
    },
    "games": {
        "raw_dir": ROOT / "data" / "raw" / "games",
        "gt_filename": "ground_truth_real.json",
        "items_meta_filename": "games_filtered.jsonl",
        "items_emb_filename": "games/games_embeddings_qwen.npy",
        "items_idx_filename": "games/games_id_to_idx_qwen.json",
        "canonical_users_filename": "canonical_test_users_games.json",
        "canonical_val_filename": "canonical_val_users_games.json",
        "canonical_train_filename": "canonical_train_users_games.json",
        "category_depth": 2,
    },
}

# Movies as SOURCE (second source scenario): the history is the user's movies, the target the
# existing music / games catalogs. Users: the existing movies x target overlap with >= 5 items in
# each (``acw.data.movies_source``); items, embeddings and metadata are the existing files.
for _t, _m in (("music", "albums_filtered.jsonl"), ("games", "games_filtered.jsonl")):
    _DOMAIN_DEFAULTS[f"movies2{_t}"] = {
        **_DOMAIN_DEFAULTS[_t],
        "raw_dir": ROOT / "data" / "raw" / f"movies2{_t}",
        "items_meta_filename": str(_DOMAIN_DEFAULTS[_t]["raw_dir"] / _m),
        "canonical_users_filename": f"canonical_test_users_movies2{_t}.json",  # unused (legacy)
        "canonical_val_filename": f"canonical_val_users_movies2{_t}.json",
        "canonical_train_filename": f"canonical_train_users_movies2{_t}.json",
        "source_idx_path": PROCESSED / "movies" / "movies_id_to_idx_qwen.json",
        "source_meta_path": ROOT / "data" / "raw" / "movies" / "movies_filtered.jsonl",
    }

KNOWN_DOMAINS: frozenset[str] = frozenset(d for d in _DOMAIN_DEFAULTS if not d.startswith("movies2"))

# ─────────────────────────────────────────────────────────────────────────────
# KNOWN_METHODS per domain
# ─────────────────────────────────────────────────────────────────────────────
#
# Entry format: (label, profiles_rel_path, index_rel_path, items_emb_rel_path,
# warn), with paths relative to PROCESSED. ``items_emb_rel_path=None`` means
# the domain's default (raw) item embeddings. ``warn`` is logged before the
# method is evaluated.
#
# Profile arrays cover all eligible users (train + val); evaluate_method()
# only reads the rows of the evaluation users, so the reported number is
# always the validation-split number.
#
# [LEAKAGE] marks results that are invalid due to structural leakage and must
# not be reported.

MethodEntry = tuple[str, str, "str | None", "str | None", "str | None"]

KNOWN_METHODS: dict[str, list[MethodEntry]] = {
    "music": [
        # ── Heuristic baselines ──────────────────────────────────────────────
        # H2: mean-pooled book profile against raw album embeddings.
        ("H2", "user_profiles_h2.npy", None, None, None),
        # H3/H4: LLM-generated profiles against whitened album embeddings.
        (
            "H3",
            "user_profiles_h3.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4",
            "user_profiles_h4.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── ZCA whitening fitted on user profiles ────────────────────────────
        (
            "H3+ZCA-profile",
            "alignment/user_profiles_h3_profile_whitened.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+ZCA-profile",
            "alignment/user_profiles_h4_profile_whitened.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H2+ZCA-profile",
            "alignment/user_profiles_h2_profile_whitened.npy",
            "user_profile_index_h2.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Linear parametric whitening (PW) ─────────────────────────────────
        (
            "H3+PW",
            "alignment/user_profiles_h3_pw.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+PW",
            "alignment/user_profiles_h4_pw.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Closed-vocabulary ("concrete") prompt variants of H3/H4 ──────────
        (
            "H3-concrete",
            "user_profiles_h3_concrete__concrete_llama70b_20260629.npy",
            "user_profile_index_h3_concrete__concrete_llama70b_20260629.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4-concrete",
            "user_profiles_h4_concrete__concrete_llama70b_20260629.npy",
            "user_profile_index_h4_concrete__concrete_llama70b_20260629.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── C2DSR-style Infomax alignment ────────────────────────────────────
        (
            "H3+Infomax",
            "alignment/user_profiles_h3_infomax.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+Infomax",
            "alignment/user_profiles_h4_infomax.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # Infomax without the ZCA-profile input stage (--no-zca).
        (
            "H3+Infomax-noZCA",
            "alignment/user_profiles_h3_infomax_noZCA.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+Infomax-noZCA",
            "alignment/user_profiles_h4_infomax_noZCA.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Non-linear PW (MLP bottleneck) ───────────────────────────────────
        (
            "H3+NL-PW",
            "alignment/user_profiles_h3_nlpw.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+NL-PW",
            "alignment/user_profiles_h4_nlpw.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Mixture-of-experts PW ────────────────────────────────────────────
        (
            "H3+MoE-PW",
            "alignment/user_profiles_h3_moe.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+MoE-PW",
            "alignment/user_profiles_h4_moe.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Multi-positive PW with PageRank anchors (invalid: GT leakage) ────
        (
            "H3+MP-PW [LEAKAGE]",
            "alignment/user_profiles_h3_mpw.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            "WARNING: INVALID RESULT — structural leakage. PageRank anchors are "
            "derived from the evaluated ground truth, so W1 learns a static "
            "projection towards popularity hubs. Do not report.",
        ),
        (
            "H4+MP-PW [LEAKAGE]",
            "alignment/user_profiles_h4_mpw.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            "WARNING: INVALID RESULT — structural leakage. See H3+MP-PW.",
        ),
        # ── Two-stage: ZCA fitted on books (ZB), then linear PW ──────────────
        (
            "H3+PW-ZB",
            "alignment/user_profiles_h3_pw_zb.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+PW-ZB",
            "alignment/user_profiles_h4_pw_zb.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Two-stage: ZCA-book, then non-linear PW ──────────────────────────
        (
            "H3+NL-PW-ZB",
            "alignment/user_profiles_h3_nlpw_zb.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+NL-PW-ZB",
            "alignment/user_profiles_h4_nlpw_zb.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Two-stage: ZCA-book, then MoE-PW ─────────────────────────────────
        (
            "H3+MoE-ZB",
            "alignment/user_profiles_h3_moe_zb.npy",
            "user_profile_index_h3.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+MoE-ZB",
            "alignment/user_profiles_h4_moe_zb.npy",
            "user_profile_index_h4.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── Multi-positive InfoNCE with catalog subgenre centroids as anchors ─
        (
            "H3+sg-PW-ZB",
            "alignment/user_profiles_h3_sg_pw_zb.npy",
            "alignment/user_profile_index_h3_sg_pw_zb.json",
            "alignment/album_embeddings_whitened.npy",
            "uses its own internal split (19,207/4,801) — not comparable to the canonical val split (2,731)",
        ),
        # ── Asymmetric whitening: raw H2 profiles vs whitened albums ─────────
        (
            "H2+albuns-whitened",
            "user_profiles_h2.npy",
            "user_profile_index_h2.json",
            "alignment/album_embeddings_whitened.npy",
            None,
        ),
        # ── H5 = L2_norm(α·H2 + (1−α)·H3+PW-ZB); best α=1.0 reduces to H2 ────
        (
            "H5(α=1.00, obsoleto≡H2)",
            "user_profiles_h5_alpha100.npy",
            "user_profile_index_h5_alpha100.json",
            "alignment/album_embeddings_whitened.npy",
            "exported only for the legacy 2,000-user sample — n reflects its "
            "intersection with the val split, not the full val split (2,731)",
        ),
    ],
    "movies": [
        # ── Heuristic baselines ──────────────────────────────────────────────
        ("H2", "user_profiles_h2_movies.npy", "user_profile_index_h2_movies.json", None, None),
        ("H3", "user_profiles_h3.npy", "user_profile_index_h3.json", None, None),
        ("H4", "user_profiles_h4.npy", "user_profile_index_h4.json", None, None),
        # ── H2/H3/H4 against whitened item embeddings ────────────────────────
        (
            "H2 (vs whitened)",
            "user_profiles_h2_movies.npy",
            "user_profile_index_h2_movies.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H3 (vs whitened)",
            "user_profiles_h3.npy",
            "user_profile_index_h3.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H4 (vs whitened)",
            "user_profiles_h4.npy",
            "user_profile_index_h4.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        # ── Linear PW ───────────────────────────────────────────────────────
        (
            "H3+PW",
            "movies/alignment/user_profiles_h3_pw_movies.npy",
            "user_profile_index_h3.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+PW",
            "movies/alignment/user_profiles_h4_pw_movies.npy",
            "user_profile_index_h4.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        # ── ZCA-book + linear PW ─────────────────────────────────────────────
        (
            "H3+PW-ZB",
            "movies/alignment/user_profiles_h3_pw_zb_movies.npy",
            "user_profile_index_h3.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+PW-ZB",
            "movies/alignment/user_profiles_h4_pw_zb_movies.npy",
            "user_profile_index_h4.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        # ── Non-linear PW (with and without ZCA-book) ────────────────────────
        (
            "H3+NL-PW",
            "movies/alignment/user_profiles_h3_nlpw_movies.npy",
            "user_profile_index_h3.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+NL-PW",
            "movies/alignment/user_profiles_h4_nlpw_movies.npy",
            "user_profile_index_h4.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H3+NL-PW-ZB",
            "movies/alignment/user_profiles_h3_nlpw_zb_movies.npy",
            "user_profile_index_h3.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+NL-PW-ZB",
            "movies/alignment/user_profiles_h4_nlpw_zb_movies.npy",
            "user_profile_index_h4.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        # ── Mixture-of-experts PW (with and without ZCA-book) ────────────────
        (
            "H3+MoE-PW",
            "movies/alignment/user_profiles_h3_moe_movies.npy",
            "user_profile_index_h3.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+MoE-PW",
            "movies/alignment/user_profiles_h4_moe_movies.npy",
            "user_profile_index_h4.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H3+MoE-ZB",
            "movies/alignment/user_profiles_h3_moe_zb_movies.npy",
            "user_profile_index_h3.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+MoE-ZB",
            "movies/alignment/user_profiles_h4_moe_zb_movies.npy",
            "user_profile_index_h4.json",
            "movies/alignment/movies_embeddings_whitened.npy",
            None,
        ),
    ],
    "games": [
        # ── Heuristic baselines ──────────────────────────────────────────────
        ("H2", "user_profiles_h2_games.npy", "user_profile_index_h2_games.json", None, None),
        ("H3", "user_profiles_h3.npy", "user_profile_index_h3.json", None, None),
        ("H4", "user_profiles_h4.npy", "user_profile_index_h4.json", None, None),
        # ── H2/H3/H4 against whitened item embeddings ────────────────────────
        (
            "H2 (vs whitened)",
            "user_profiles_h2_games.npy",
            "user_profile_index_h2_games.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H3 (vs whitened)",
            "user_profiles_h3.npy",
            "user_profile_index_h3.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H4 (vs whitened)",
            "user_profiles_h4.npy",
            "user_profile_index_h4.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        # ── Linear PW ───────────────────────────────────────────────────────
        (
            "H3+PW",
            "games/alignment/user_profiles_h3_pw_games.npy",
            "user_profile_index_h3.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+PW",
            "games/alignment/user_profiles_h4_pw_games.npy",
            "user_profile_index_h4.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        # ── ZCA-book + linear PW ─────────────────────────────────────────────
        (
            "H3+PW-ZB",
            "games/alignment/user_profiles_h3_pw_zb_games.npy",
            "user_profile_index_h3.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+PW-ZB",
            "games/alignment/user_profiles_h4_pw_zb_games.npy",
            "user_profile_index_h4.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        # ── Non-linear PW (with and without ZCA-book) ────────────────────────
        (
            "H3+NL-PW",
            "games/alignment/user_profiles_h3_nlpw_games.npy",
            "user_profile_index_h3.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+NL-PW",
            "games/alignment/user_profiles_h4_nlpw_games.npy",
            "user_profile_index_h4.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H3+NL-PW-ZB",
            "games/alignment/user_profiles_h3_nlpw_zb_games.npy",
            "user_profile_index_h3.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+NL-PW-ZB",
            "games/alignment/user_profiles_h4_nlpw_zb_games.npy",
            "user_profile_index_h4.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        # ── Mixture-of-experts PW (with and without ZCA-book) ────────────────
        (
            "H3+MoE-PW",
            "games/alignment/user_profiles_h3_moe_games.npy",
            "user_profile_index_h3.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+MoE-PW",
            "games/alignment/user_profiles_h4_moe_games.npy",
            "user_profile_index_h4.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H3+MoE-ZB",
            "games/alignment/user_profiles_h3_moe_zb_games.npy",
            "user_profile_index_h3.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
        (
            "H4+MoE-ZB",
            "games/alignment/user_profiles_h4_moe_zb_games.npy",
            "user_profile_index_h4.json",
            "games/alignment/games_embeddings_whitened.npy",
            None,
        ),
    ],
}


# ─────────────────────────────────────────────────────────────────────────────
# resolve_paths
# ─────────────────────────────────────────────────────────────────────────────


def resolve_paths(
    domain: str = "music",
    *,
    gt_path: str | None = None,
    items_emb_path: str | None = None,
    items_idx_path: str | None = None,
    canonical_users_path: str | None = None,
    canonical_val_path: str | None = None,
    canonical_train_path: str | None = None,
    items_meta_path: str | None = None,
) -> DomainConfig:
    """Resolve all paths for an evaluation domain.

    ``domain`` sets the defaults for every field; any explicit argument
    overrides its field individually.

    Args:
        domain: target domain — ``"music"``, ``"movies"`` or ``"games"``.
        gt_path: ground-truth JSON.
        items_emb_path: target item embeddings (.npy).
        items_idx_path: item id → row index mapping (.json).
        canonical_users_path: legacy fixed list of evaluation users.
        canonical_val_path: validation-split user list.
        canonical_train_path: training-split user list.
        items_meta_path: item metadata (.jsonl), used for subgenre expansion.

    Returns:
        A fully resolved ``DomainConfig``.
    """
    if domain not in _DOMAIN_DEFAULTS:
        raise ValueError(f"Unknown domain: {domain!r}. Valid: {sorted(_DOMAIN_DEFAULTS)}")

    d = _DOMAIN_DEFAULTS[domain]
    raw_dir: Path = d["raw_dir"]

    def _p(override: str | None, default: Path) -> Path:
        return Path(override) if override is not None else default

    return DomainConfig(
        domain=domain,
        raw_dir=raw_dir,
        gt_path=_p(gt_path, raw_dir / d["gt_filename"]),
        items_meta_path=_p(items_meta_path, raw_dir / d["items_meta_filename"]),
        items_emb_path=_p(items_emb_path, PROCESSED / d["items_emb_filename"]),
        items_idx_path=_p(items_idx_path, PROCESSED / d["items_idx_filename"]),
        canonical_users_path=_p(canonical_users_path, PROCESSED / d["canonical_users_filename"]),
        canonical_val_path=_p(canonical_val_path, PROCESSED / d["canonical_val_filename"]),
        canonical_train_path=_p(canonical_train_path, PROCESSED / d["canonical_train_filename"]),
        category_depth=d["category_depth"],
        source_idx_path=d.get("source_idx_path", PROCESSED / "book_id_to_idx.json"),
        source_meta_path=d.get("source_meta_path"),
    )
