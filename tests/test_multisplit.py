import pandas as pd

from experiments.granularity.multisplit import SPEC_IDS, SPECS, summarise, verdict


def test_verdict_uses_holm_p_and_the_sign():
    assert verdict(0.01, 1e-3) == "better" and verdict(0.01, -1e-3) == "worse"
    assert verdict(0.2, 1e-3) == "ns" and verdict(float("nan"), 1e-3) == "ns"


def test_summary_counts_splits_and_takes_the_sample_sd():
    rows = [("a1", "music", s, d, p) for s, d, p in
            [(42, 2.0, 0.001), (43, 3.0, 0.2), (44, 1.0, 0.04)]]  # fmt: skip
    df = pd.DataFrame(rows, columns=["contrast", "domain", "seed", "mean_diff", "p_holm_signflip"])
    df["label"], df["n_nonzero_diff"] = "x", 100
    df["verdict"] = [verdict(p, d) for p, d in zip(df["p_holm_signflip"], df["mean_diff"])]
    r = summarise(df).iloc[0]
    assert (r["n_splits"], r["n_better"], r["n_ns"], r["n_worse"]) == (3, 2, 1, 0)
    assert r["mean_diff_mean"] == 2.0 and r["mean_diff_sd"] == 1.0


def test_contrast_ids_are_unique():
    assert len(SPEC_IDS) == len(set(SPEC_IDS)) == len(SPECS)


def test_centring_share_per_split():
    from experiments.granularity.multisplit import centring_share

    rows = [(c, "music", s, d) for s, (u, c0) in enumerate([(4.0, 2.0), (4.0, 3.0)])
            for c, d in (("qU_vs_raw", u), ("qCtr_vs_raw", c0))]  # fmt: skip
    df = pd.DataFrame(rows, columns=["contrast", "domain", "seed", "mean_diff"])
    r = centring_share(df).iloc[0]
    assert (r["share_mean"], r["share_min"], r["share_max"]) == (0.625, 0.5, 0.75)
