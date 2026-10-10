import pytest

from experiments.figures import paper_tables as pt


def test_signed_uses_math_minus_and_hides_negative_zero():
    assert pt.signed(-40.4) == "$-$40" and pt.signed(157.6) == "$+$158"
    assert pt.signed(-0.2) == "0" and pt.signed(float("nan")) == "--"


def test_small_and_p_formatting():
    assert (
        pt.small(0.0) == "0"
        and pt.small(5.9e-05) == r"$\approx 0$"
        and pt.small(0.0184, 3) == "0.018"
    )
    assert pt.ptex(1e-9) == "$p<0.001$" and pt.ptex(0.081) == "$p=0.081$"


def test_mark_needs_holm_significance_and_follows_the_sign():
    assert pt.mark(0.01, 1.0) == pt.UP and pt.mark(0.01, -1.0) == pt.DOWN
    assert pt.mark(0.2, 1.0) == "" and pt.mark(float("nan"), 1.0) == ""


def test_tex_escapes_specials_and_thousands_is_math_safe():
    assert pt.tex("R&B_50%") == r"R\&B\_50\%" and pt.thousands(4884) == "4{,}884"


@pytest.mark.skipif(not (pt.GRAN / "summary_test.csv").exists(), reason="result files absent")
def test_encoder_table_counts_and_rows():
    R = pt.Results()
    text = pt.table_encoders_gap(R)
    assert text.count("\\\\") >= 14 and "nan" not in text
    # query ZCA gains by item with Holm significance in all 10 music encoders (8 verified)
    assert "10 / 0 / 0" in text and "8 / 0 / 0" in text
    for enc in pt.ENCODERS:
        assert pt.ENCODERS[enc][0] in text


@pytest.mark.skipif(not (pt.GRAN / "stats_hybrid.csv").exists(), reason="hybrid results absent")
def test_hybrid_tables_cover_the_three_domains():
    text = pt.table_hybrid()
    assert "nan" not in text and text.count("+ popularity") == 9
    assert all(pt.DNAME[d] in text for d in pt.DOMAINS)
    val = pt.table_hybrid_val()
    assert val.count("\\textbf") == 9 and "nan" not in val


@pytest.mark.skipif(not (pt.GRAN / "stats_query_norm.csv").exists(), reason="results absent")
def test_query_norm_tables_report_discordant_users_and_the_verified_tally():
    R = pt.Results()
    tests = pt.table_qnorm_tests(R)
    assert "nan" not in tests and tests.count("<0.001") >= 6
    assert "8 / 0 / 0" in tests  # query ZCA beats catalog ZCA in the 8 verified encoders
    enc = pt.table_encoders(R)
    assert enc.count("&") > 100 and "8 / 0 / 0" in enc and "nan" not in enc
    for name in (pt.ENCODERS[e][0] for e in pt.VERIFIED_MUSIC):
        assert name in enc
    q = pt.table_qnorm(R)
    assert "nan" not in q and q.count("\\multicolumn{4}{c}") >= 6


@pytest.mark.skipif(not (pt.GRAN / "stats_knn.csv").exists(), reason="results absent")
def test_lexical_controls_knn_and_cold_tables_build_without_nan():
    R = pt.Results()
    for fn in (pt.table_lexical, pt.table_controls, pt.table_coldwarm, pt.table_knn,
               pt.table_text_format, pt.table_alpha, pt.table_tally):  # fmt: skip
        assert "nan" not in fn(R).lower().replace("tabular", ""), fn.__name__
    assert pt.arm_label("zca_train_users_a0.75") == r"ZCA $\alpha{=}0.75$ (users)"
    assert pt.arm_label("zcalw_train_users") == "shrunk ZCA (users)"
