"""Selection of the users for whom H2 profiles are generated."""

import json
import random

from .config import N_TEST_USERS, PROCESSED_DIR, RAW_DIR, SEED


def load_all_users(domain: str = "music") -> list[str]:
    if domain == "music":
        path = RAW_DIR / "overlapping_users.json"
    else:
        path = RAW_DIR / domain / "overlapping_users.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    users = list(data.keys()) if isinstance(data, dict) else list(data)
    print(f"Loaded {len(users):,} overlapping users (all_users=True, domain={domain})")
    return users


def sample_test_users(history: dict) -> list[str]:
    """Return the canonical test users.

    Source of truth: data/processed/canonical_test_users.json. If that file
    does not exist, users are sampled with random.seed(SEED) and the file is
    written.
    """
    canonical_path = PROCESSED_DIR / "canonical_test_users.json"

    if canonical_path.exists():
        with open(canonical_path) as f:
            users = json.load(f)
        # Keep only users with a source-domain history.
        users = [u for u in users if u in history]
        print(f"Eligible users (genre GT valid): {len(users):,}")
        return users

    # ── Fallback: sample and save the canonical user list ─────────────────
    gt_path = RAW_DIR / "ground_truth_music.json"
    if not gt_path.exists():
        gt_path = RAW_DIR / "ground_truth_real.json"
    with open(gt_path, encoding="utf-8") as f:
        gt = json.load(f)

    album_to_genre = {}
    with open(RAW_DIR / "albums_filtered.jsonl", encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            asin = item.get("parent_asin")
            cats = item.get("categories", [])
            if not asin or len(cats) < 2:
                continue
            genre = cats[2] if len(cats) >= 3 else cats[1]
            album_to_genre[asin] = genre

    MAX_GENRES = 6
    eligible = []
    for uid in history.keys():
        albums = gt.get(uid, [])
        genres = {album_to_genre.get(a) for a in albums if a in album_to_genre}
        genres.discard(None)
        if genres and len(genres) <= MAX_GENRES:
            eligible.append(uid)

    print(f"Eligible users (genre GT valid): {len(eligible):,}")
    random.seed(SEED)
    sampled = random.sample(eligible, min(N_TEST_USERS, len(eligible)))

    # Persist for later runs.
    with open(canonical_path, "w") as f:
        json.dump(sampled, f)
    print(f"Saved canonical test users to {canonical_path}")

    return sampled
