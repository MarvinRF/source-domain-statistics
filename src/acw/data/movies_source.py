"""Users and files of the pairs with MOVIES as the source domain (``movies2music``, ``movies2games``).

The derived movies history (``data/raw/movies/ground_truth_real.json``: the user's reviewed movies
with rating >= 4, items 5-core, ordered by timestamp) is the source history; the existing ground
truth of the target is the relevance. A user is kept with at least ``MIN_ITEMS`` (5) items in each
(the rule of docs/DATA.md). Nothing is rescanned from the raw dumps.

Output: ``data/raw/movies2{target}/{user_book_history,ground_truth_real}.json``.

Usage::

    python -m acw.data.movies_source
"""

from __future__ import annotations

import json

from acw.settings import RAW_DIR

MIN_ITEMS = 5
TARGETS = {
    "music": RAW_DIR / "ground_truth_real.json",
    "games": RAW_DIR / "games" / "ground_truth_real.json",
}


def build(target: str) -> dict:
    movies = json.loads((RAW_DIR / "movies" / "ground_truth_real.json").read_text())
    gt = json.loads(TARGETS[target].read_text())
    users = sorted(
        u for u in movies.keys() & gt.keys() if min(len(movies[u]), len(gt[u])) >= MIN_ITEMS
    )
    out = RAW_DIR / f"movies2{target}"
    out.mkdir(exist_ok=True)
    (out / "user_book_history.json").write_text(json.dumps({u: movies[u] for u in users}))
    (out / "ground_truth_real.json").write_text(json.dumps({u: gt[u] for u in users}))
    return {"target": target, "n_users": len(users), "min_items": MIN_ITEMS}


if __name__ == "__main__":
    for t in TARGETS:
        print(json.dumps(build(t)))
