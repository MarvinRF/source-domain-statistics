import numpy as np
import pandas as pd

from experiments.granularity.stats import EXCLUDED, run_stems, tallies


def _rows(encoder, p, d, conditions):
    return [
        {
            "domain": "music",
            "encoder": encoder,
            "gt": "exact",
            "metric": "ndcg10",
            "condition": c,
            "p_holm_signflip": p,
            "mean_diff": d,
        }
        for c in conditions
    ]


def test_tallies_skip_pca_and_baselines_and_split_verified_from_unverified():
    corr = ["center_catalog", "zca_catalog_a1.0", EXCLUDED[0]]
    vs_raw = pd.DataFrame(
        _rows(
            "gte", 0.01, 1.0, corr + ["popularity", "random"]
        )  # 2 better (+ pca, baselines ignored)
        + _rows("gte", 0.5, 1.0, ["abtt_catalog_D1"])  # 1 not different
        + _rows("qwen3", 0.01, -1.0, corr)  # unverified: 2 worse
        + [{**_rows("gte", np.nan, 1.0, ["x"])[0]}]  # NaN p (outside any family) counts as ns
    )
    t = tallies(vs_raw, vs_raw.iloc[:0], vs_raw.iloc[:0])
    r = t[(t["reference"] == "raw") & (t["gt"] == "all")].set_index("encoders")
    assert r.loc["verified", ["n_tests", "better", "worse", "ns"]].tolist() == [4, 2, 0, 2]
    assert r.loc["unverified", ["n_tests", "better", "worse", "ns"]].tolist() == [2, 0, 2, 0]


def test_run_stems_keeps_only_grid_files(tmp_path, monkeypatch):
    import experiments.granularity.stats as st

    for name in ("music_gte_titlecat", "hybrid_music_gte", "knn_music_gte", "lexical_music_gte",
                 "games_bge-large_titlecat", "music_gte_titlecat_legacy", "query_norm_music_gte"):  # fmt: skip
        (tmp_path / f"{name}.csv.gz").write_text("")
    monkeypatch.setattr(st, "OUT_DIR", tmp_path)
    assert run_stems() == ["games_bge-large_titlecat", "music_gte_titlecat"]
