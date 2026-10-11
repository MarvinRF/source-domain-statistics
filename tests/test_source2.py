from pathlib import Path

import numpy as np
import pytest

from acw.data.splits import DOMAINS as SPLIT_DOMAINS
from acw.evaluation.config import KNOWN_DOMAINS, resolve_paths
from experiments.granularity import common as C
from experiments.granularity.bridge import fit_mlp
from experiments.granularity.run_corrections import REGISTRY


def test_movies_source_pairs_are_registered_without_touching_the_books_pairs():
    for d in C.DOMAINS2:
        src = REGISTRY[(d, "gte", "titlecat")]
        assert src.books == "movies/movies_embeddings_qwen.npy"  # "books" = source catalog
        assert src.items == REGISTRY[(d.removeprefix("movies2"), "gte", "titlecat")].items
        cfg = resolve_paths(d)
        assert cfg.source_idx_path.name == "movies_id_to_idx_qwen.json"
        assert cfg.source_meta_path.name == "movies_filtered.jsonl"
        assert cfg.raw_dir.name == d and d in SPLIT_DOMAINS
    assert KNOWN_DOMAINS == {"music", "movies", "games"}
    assert resolve_paths("music").source_idx_path.name == "book_id_to_idx.json"
    assert all(d in C.DOMAINS for d, _, _ in C.verified_runs())  # books --stats unchanged
    assert {d for d, _, _ in C.verified_runs(C.DOMAINS2)} == set(C.DOMAINS2)
    assert [d for d, _ in C.gte_runs(C.DOMAINS2)] == list(C.DOMAINS2)


def test_mlp_bridge_shapes_and_learns_a_nonlinear_map():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(600, 8)).astype(np.float32)
    Y = np.tanh(X @ rng.normal(size=(8, 5))).astype(np.float32)
    f, ep = fit_mlp(X[:500], Y[:500], X[500:], Y[500:], width=32, wd=0.0)
    out = f(X[500:])
    assert out.shape == (100, 5) and out.dtype == np.float32 and ep >= 1
    assert ((out - Y[500:]) ** 2).mean() < Y[500:].var(0).mean()  # better than the mean


@pytest.mark.skipif(not (Path(__file__).parents[1] / "data/splits/music_stats.json").exists(), reason="needs data/splits (make splits-new)")
def test_source2_table_and_macros_render_from_the_committed_results():
    from experiments.figures import paper_numbers as pn
    from experiments.figures import paper_tables as pt

    R = pt.Results()
    for tab in (pt.table_source2(R), pt.table_bridge(R)):
        assert "nan" not in tab and "Movies$\\to$Music" in tab
    pn.NUMS.clear()
    pn.build()
    assert "STwoQUDiffMovMus" in pn.NUMS and "BrMlpRawDiffMusic" in pn.NUMS
