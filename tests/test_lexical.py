import numpy as np

from experiments.granularity.lexical import (
    K1,
    B,
    LexIndex,
    lexical_top,
    strict_tokens,
    tokens,
)

DOCS = ["red fox", "red red dog", "blue sky", "red"]


def _index(**kw):
    return LexIndex([tokens(d) for d in DOCS], **kw)


def test_tokens_split_on_non_alphanumerics_and_lowercase():
    assert tokens("NBA 2K17 - Early Tip-Off_Edition!") == [
        "nba",
        "2k17",
        "early",
        "tip",
        "off",
        "edition",
    ]
    assert strict_tokens("The Big Apple of 2K17 music") == ["apple"]  # >= 4 letters, stop list


def test_bm25_matches_the_textbook_formula_and_counts_the_query_term_frequency():
    idx = _index()
    n, df = 4, 3  # "red" is in docs 0, 1, 3
    idf = np.log1p((n - df + 0.5) / (df + 0.5))
    avgdl = np.mean([len(tokens(d)) for d in DOCS])

    def w(tf, dl):
        return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))

    s = idx.scores("bm25", idx.query([tokens("red")]))[0]
    np.testing.assert_allclose(s, [w(1, 2), w(2, 3), 0.0, w(1, 1)])
    s2 = idx.scores("bm25", idx.query([tokens("red red")]))[0]
    np.testing.assert_allclose(s2, 2 * s)


def test_idf_overlap_counts_each_shared_token_once_and_ignores_unknown_tokens():
    idx = _index()
    q = idx.query([tokens("red red fox zzz")])
    s = idx.scores("idf_overlap", q)[0]
    idf_red = np.log1p((4 - 3 + 0.5) / (3 + 0.5))
    idf_fox = np.log1p((4 - 1 + 0.5) / (1 + 0.5))
    np.testing.assert_allclose(s, [idf_red + idf_fox, idf_red, 0.0, idf_red])


def test_strict_idf_is_log_n_over_one_plus_df():
    idx = LexIndex(
        [strict_tokens(d) for d in ["apple pie", "apple tart", "plum cake"]], strict_idf=True
    )
    s = idx.scores("idf_overlap", idx.query([["apple"]]))[0]
    np.testing.assert_allclose(s, [np.log(3 / 3)] * 2 + [0.0], atol=1e-12)


def test_lexical_top_orders_by_score_then_by_tie_rank():
    idx = _index()
    tie_rank = np.array([3, 2, 1, 0])
    top = lexical_top(idx, "idf_overlap", [tokens("red"), tokens("zzz")], tie_rank, k=4)
    assert top[0].tolist() == [3, 1, 0, 2]  # docs with "red" (equal scores) by rank, then the rest
    assert top[1].tolist() == [3, 2, 1, 0]  # no match: by tie rank only
