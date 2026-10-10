import json

from acw.evaluation.config import DomainConfig
from acw.evaluation.pipeline import train_popularity, trivial_baselines, trivial_rankings


def _cfg(tmp_path):
    items = [
        {"parent_asin": "a1", "categories": ["CDs", "Rock", "Indie"], "rating_number": 5},
        {"parent_asin": "a2", "categories": ["CDs", "Rock", "Indie"], "rating_number": 9},
        {"parent_asin": "a3", "categories": ["CDs", "Jazz", "Fusion"], "rating_number": 7},
    ]
    (tmp_path / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    gt = {"t1": ["a3"], "t2": ["a3"], "t3": ["a1"], "e1": ["a1"], "e2": ["a1"]}
    (tmp_path / "gt.json").write_text(json.dumps(gt))
    p = tmp_path
    return DomainConfig(
        domain="music",
        raw_dir=p,
        gt_path=p / "gt.json",
        items_meta_path=p / "items.jsonl",
        items_emb_path=p / "x.npy",
        items_idx_path=p / "idx.json",
        canonical_users_path=p / "t.json",
        canonical_val_path=p / "v.json",
        canonical_train_path=p / "tr.json",
        category_depth=2,
    )


IDX = {0: "a1", 1: "a2", 2: "a3"}


def test_top_category_is_chosen_from_train_users_only(tmp_path):
    cfg = _cfg(tmp_path)
    ranks, name = trivial_rankings(cfg, IDX, ["t1", "t2", "t3"])
    assert name == "Fusion"  # 2 train users vs 1; the eval users e1/e2 favour Indie
    assert ranks["top_category"] == ["a3"]
    assert ranks["popularity"] == ["a3", "a1", "a2"]


def test_popularity_counts_train_users_only(tmp_path):
    cfg = _cfg(tmp_path)
    assert train_popularity(cfg, ["t1", "t2", "t3"]) == {"a3": 2, "a1": 1}  # e1/e2 not counted
    # rating_number (a2 highest) ignores who rated; the train count does not
    legacy = trivial_rankings(cfg, IDX, ["t1", "t2", "t3"], legacy_popularity=True)[0]
    assert legacy["popularity"] == ["a2", "a3", "a1"]
    # equal counts: ties are broken by item id
    assert trivial_rankings(cfg, IDX, ["t1", "t3"])[0]["popularity"] == ["a1", "a3", "a2"]


def test_without_train_users_no_baselines(tmp_path):
    cfg = _cfg(tmp_path)  # canonical train file does not exist
    assert trivial_baselines(cfg, {"e1": {"a1"}}, ["e1"], IDX) == []


def test_explicit_train_users_give_both_rows(tmp_path):
    cfg = _cfg(tmp_path)
    rows = trivial_baselines(cfg, {"e1": {"a1"}}, ["e1"], IDX, train_users=["t1", "t2"])
    assert [r["label"] for r in rows] == ["Popularity", "Top-genre (Fusion)"]


def test_top_category_depends_on_depth(tmp_path):
    cfg = _cfg(tmp_path)
    assert trivial_rankings(cfg, IDX, ["t1", "t2", "t3"], depth=1)[1] == "Jazz"
    assert trivial_rankings(cfg, IDX, ["t1", "t2", "t3"], depth=2)[1] == "Fusion"
