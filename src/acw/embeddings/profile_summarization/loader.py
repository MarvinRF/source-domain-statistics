"""Loading of evaluation users, user histories, item metadata and intent entities."""

import json
import logging
import random
from pathlib import Path

from .config import (
    N_TEST_USERS,
    PROCESSED_DIR,
    RAW_DIR,
    SEED,
)

log = logging.getLogger(__name__)


def domain_raw_dir(domain: str | None) -> Path:
    """Raw-data directory for a target domain.

    Returns data/raw/ for music (domain=None, the default books -> CDs pair)
    or data/raw/{domain}/ for movies/games. Each domain has its own
    overlapping_users.json, user_book_history.json and books_filtered.jsonl
    with a different user population (movies: 67,458 overlapping users,
    games: 9,840, music: 24,420), so mixing them up silently generates
    profiles for the wrong population."""
    if domain is None:
        return RAW_DIR
    return RAW_DIR / domain


def domain_from_heuristic(heuristic: str | None) -> str | None:
    if not heuristic:
        return None
    if heuristic.endswith("_movies"):
        return "movies"
    if heuristic.endswith("_games"):
        return "games"
    return None


def load_test_users(
    seed: int = SEED,
    n: int = N_TEST_USERS,
    all_users: bool = False,
    user_list_path: str | None = None,
    domain: str | None = None,
) -> list[str]:
    """Return the user IDs to generate profiles for.

    Args:
        domain: None (music, default) | "movies" | "games". Selects
            data/raw/{domain}/overlapping_users.json instead of
            data/raw/overlapping_users.json, since each domain has a
            different overlapping population.
        user_list_path: explicit JSON list of user_ids (e.g.
            data/processed/canonical_val_users.json, the canonical
            validation population, n=2,731). Takes precedence over
            all_users and the default.
        all_users: if False (default), return the 2,000 canonical test
            users from canonical_test_users.json; if True, return all
            overlapping users (for large-scale train + test generation).
    """
    if user_list_path:
        with open(user_list_path, encoding="utf-8") as f:
            data = json.load(f)
        users = list(data.keys()) if isinstance(data, dict) else list(data)
        log.info(f"  Loaded {len(users):,} users from explicit list: {user_list_path}")
        return users

    if all_users:
        raw_dir = domain_raw_dir(domain)
        overlapping_path = raw_dir / "overlapping_users.json"
        with open(overlapping_path, encoding="utf-8") as f:
            data = json.load(f)
        # overlapping_users.json may be a list or a dict {uid: ...}
        users = list(data.keys()) if isinstance(data, dict) else list(data)
        log.info(
            f"  Loaded {len(users):,} overlapping users (all_users=True, "
            f"domain={domain or 'music'}, path={overlapping_path})"
        )
        return users

    canonical_path = PROCESSED_DIR / "canonical_test_users.json"
    if canonical_path.exists():
        with open(canonical_path, encoding="utf-8") as f:
            users = json.load(f)
        log.info(f"  Loaded {len(users):,} canonical test users")
        return users

    # Fallback: sample with random.Random(seed) and save the canonical list.
    with open(RAW_DIR / "user_book_history.json", encoding="utf-8") as f:
        history = json.load(f)
    with open(RAW_DIR / "ground_truth_real.json", encoding="utf-8") as f:
        gt = json.load(f)

    album_to_genre: dict[str, str] = {}
    with open(RAW_DIR / "albums_filtered.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            asin = item.get("parent_asin")
            cats = item.get("categories", [])
            if not asin or len(cats) < 2:
                continue
            album_to_genre[asin] = cats[2] if len(cats) >= 3 else cats[1]

    MAX_GENRES = 6
    eligible = []
    for uid in history.keys():
        albums = gt.get(uid, [])
        genres = {album_to_genre.get(a) for a in albums if a in album_to_genre}
        genres.discard(None)
        if genres and len(genres) <= MAX_GENRES:
            eligible.append(uid)

    rng = random.Random(seed)
    sampled = rng.sample(eligible, min(n, len(eligible)))

    with open(canonical_path, "w") as f:
        json.dump(sampled, f)
    log.info(f"  Saved {len(sampled):,} canonical test users to {canonical_path}")
    return sampled


def load_book_history(domain: str | None = None) -> dict:
    raw_dir = domain_raw_dir(domain)
    with open(raw_dir / "user_book_history.json", encoding="utf-8") as f:
        return json.load(f)


def load_book_metadata(domain: str | None = None) -> dict:
    raw_dir = domain_raw_dir(domain)
    log.info(f"Loading book metadata (domain={domain or 'music'})...")
    meta: dict = {}
    with open(raw_dir / "books_filtered.jsonl", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            asin = item.get("parent_asin") or item.get("asin", "")
            title = item.get("title", "").strip()
            desc = item.get("description", "")
            if isinstance(desc, list):
                desc = " ".join(str(d) for d in desc if d)
            meta[asin] = {"title": title, "description": str(desc).strip()[:300]}
    log.info(f"  Loaded {len(meta):,} books")
    return meta


def load_book_intent_cache() -> dict:
    path = PROCESSED_DIR / "book_intent_cache.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    log.info(f"  Loaded {len(data):,} intent entities")
    return data
