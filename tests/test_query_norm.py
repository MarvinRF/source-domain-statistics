import math

import numpy as np
import pandas as pd

from acw.corrections import fit_moments, zca_power
from experiments.granularity import common as C
from experiments.granularity.query_norm import (
    A_PRIORI,
    CATALOG_ARMS,
    contrasts,
    matrix,
    query_arms,
    versus_raw,
)


def _unit(rng, n, d):
    x = rng.normal(size=(n, d)) @ np.diag(np.linspace(0.2, 2.0, d)) + 0.5
    return (x / np.linalg.norm(x, axis=1, keepdims=True)).astype(np.float32)


def test_shrink_keeps_the_mean_eigenvalue_and_vanishes_with_many_samples():
    rng = np.random.default_rng(0)
    small, large = _unit(rng, 30, 8), _unit(rng, 20_000, 8)
    ms, ml = fit_moments(small), fit_moments(large)
    ws, ds = C.shrink(ms, small)
    wl, dl = C.shrink(ml, large)
    assert 0 < dl < ds <= 1
    np.testing.assert_allclose(ws.eigvals.mean(), ms.eigvals.mean())
    np.testing.assert_allclose(wl.eigvals, ml.eigvals, rtol=0.05)
    assert ws.eigvals.min() > ms.eigvals.min()  # shrinkage lifts the smallest eigenvalue


def test_query_arms_cover_three_sources_and_match_the_direct_transform():
    rng = np.random.default_rng(1)
    x = _unit(rng, 200, 16)
    moments = {s: fit_moments(_unit(rng, 100, 16)) for s in C.SOURCES}
    shrunk = {s: C.shrink(m, x)[0] for s, m in moments.items()}
    arms = query_arms(moments, shrunk)
    assert len(arms) == 3 * 9
    assert {s for s, _ in arms.values()} == set(C.SOURCES)
    src, fn = arms["zca_books_catalog_a0.5"]
    np.testing.assert_allclose(fn(x), zca_power(x, *moments["books_catalog"], 0.5), atol=1e-6)
    for _, fn in arms.values():
        out = fn(x)
        assert out.shape == x.shape and np.allclose(np.linalg.norm(out, axis=1), 1, atol=1e-4)
    for name in A_PRIORI:
        assert name in arms or name in ("zca_catalog_a1.0", "zca_both_separate")


def test_contrasts_and_versus_raw_use_the_selected_arms():
    sel = {
        "query_side": "abtt_books_catalog_D3",
        "catalog_side": "zca_catalog_train_users",
        "per_source": {"train_users": "zcalw_train_users"},
    }
    tests = {t[0]: t for t in contrasts(sel)}
    assert tests["a_selected_query_vs_selected_catalog"][1:] == (
        "abtt_books_catalog_D3",
        "zca_catalog_train_users",
    )
    assert tests["c_both_sides_vs_query_only"][1:] == ("zca_both_separate", "zca_train_users_a1.0")
    names = [c for _, c, _ in versus_raw(sel)]
    assert len(names) == len(set(names)) and "zcalw_train_users" in names


def test_fuse_orders_by_aux_first_when_lambda_is_inf():
    s = np.array([[0.9, 0.1, 0.5]], dtype=np.float32)
    aux = np.array([0.0, 1.0, 1.0])
    out = C.fuse(s, aux, math.inf)
    assert np.argsort(-out[0]).tolist() == [2, 1, 0]  # aux first, s breaks the tie of 1 vs 1
    np.testing.assert_allclose(C.fuse(s, aux, 2.0), s + 2.0 * aux.astype(np.float32))
    np.testing.assert_allclose(C.fuse(s, aux, 0), s)


def test_tally_counts_verdicts_by_encoder_status():
    def row(enc, p, d):
        return {"encoder": enc, "p_holm_signflip": p, "mean_diff": d, "test": "t"}

    rows = [row("gte", 0.01, 1), row("gte", 0.01, -1), row("gte", 0.5, 1), row("qwen3", 0.01, 1)]
    t = C.tally(rows, ["test"]).set_index("encoders")
    assert t.loc["verified", ["n_tests", "better", "worse", "ns"]].tolist() == [3, 1, 1, 1]
    assert t.loc["unverified", ["n_tests", "better", "worse", "ns"]].tolist() == [1, 1, 0, 0]


def test_block_means_counts_users_with_a_hit():
    df = pd.DataFrame(
        {
            "user_id": ["a", "b", "a", "b"],
            "gt": ["exact"] * 4,
            "condition": ["x", "x", "y", "y"],
            "nDCG@10": [0.5, 0.0, 0.0, 0.0],
            "R@10": [0, 0, 0, 0],
            "R@50": [0, 0, 0, 0],
            "R@100": [1.0, 0.0, 0.0, 0.0],
        }
    )
    m = C.block_means(df)["exact"]
    assert m["x"]["n_hit10"] == 1 and m["x"]["n_hit100"] == 1 and m["x"]["n"] == 2
    assert m["y"]["n_hit10"] == 0


def test_text_variants_name_their_files_and_extend_only_the_music_gte_runs():
    assert C.text_tag("titlecat") == "" and C.text_tag("title") == "_title"
    runs = C.verified_runs()
    assert ("music", "gte", "title") in runs and ("music", "gte", "desc") in runs
    assert ("movies", "gte", "titlecat") in runs and ("music", "qwen3", "titlecat") not in runs
    assert all(t == "titlecat" for d, e, t in runs if (d, e) != ("music", "gte"))
    assert C.gte_runs()[:3] == [(d, "titlecat") for d in C.DOMAINS]
    assert ("music", "title") in C.gte_runs()


def test_matrix_family_uses_existing_arms_and_crosses_side_with_source():
    rng = np.random.default_rng(2)
    moments = {s: fit_moments(_unit(rng, 50, 8)) for s in C.SOURCES}
    arms = set(query_arms(moments, moments))
    tests = matrix()
    assert len(tests) == 13 and len({t for t, _, _ in tests}) == 13
    for _, cond, ref in tests:
        for c in (cond, ref):
            assert c == "raw" or c in arms or c in CATALOG_ARMS
    side = {t: (c, r) for t, c, r in tests if t.startswith("matrix:side_")}
    assert side["matrix:side_U"] == ("zca_train_users_a1.0", "zca_catalog_train_users")
    assert side["matrix:side_C"] == ("zca_target_catalog_a1.0", "zca_catalog_a1.0")
