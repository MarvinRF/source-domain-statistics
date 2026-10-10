import json

from acw.evaluation.config import DomainConfig
from acw.evaluation.loader import build_exact_ground_truth, build_genre_ground_truth


def _cfg(tmp_path):
    items = [
        {"parent_asin": "a1", "categories": ["CDs", "Rock", "Indie"]},
        {"parent_asin": "a2", "categories": ["CDs", "Rock", "Indie"]},
        {"parent_asin": "a3", "categories": ["CDs", "Jazz", "Fusion"]},
        {"parent_asin": "a4", "categories": ["CDs", "Pop"]},
    ]
    (tmp_path / "items.jsonl").write_text("\n".join(json.dumps(i) for i in items))
    (tmp_path / "gt.json").write_text(json.dumps({"u1": ["a1"], "u2": ["a3", "zz"], "u3": ["zz"]}))
    (tmp_path / "idx.json").write_text(json.dumps({"a1": 0, "a2": 1, "a3": 2, "a4": 3}))
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


def test_exact_is_restricted_to_catalog(tmp_path):
    gt = build_exact_ground_truth(_cfg(tmp_path))
    assert gt == {"u1": {"a1"}, "u2": {"a3"}}  # "zz" not in catalog; u3 has nothing left


def test_genre_expands_exact(tmp_path):
    cfg = _cfg(tmp_path)
    exact, genre = build_exact_ground_truth(cfg), build_genre_ground_truth(cfg)
    assert genre["u1"] == {"a1", "a2"}  # same depth-2 subgenre "Indie"
    for uid, items in exact.items():
        assert items <= genre[uid]


def test_depth_selects_the_taxonomy_level(tmp_path):
    cfg = _cfg(tmp_path)
    assert build_genre_ground_truth(cfg, depth=0)["u1"] == {"a1", "a2", "a3", "a4"}  # "CDs"
    assert build_genre_ground_truth(cfg, depth=1)["u2"] == {"a3"}  # Jazz
    assert build_genre_ground_truth(cfg, depth=2)["u1"] == {"a1", "a2"}  # Indie
    assert build_genre_ground_truth(cfg) == build_genre_ground_truth(cfg, depth=2)


def test_max_genres_filter_is_optional(tmp_path):
    cfg = _cfg(tmp_path)
    (tmp_path / "gt.json").write_text(json.dumps({"u4": ["a1", "a3"]}))
    assert build_genre_ground_truth(cfg, depth=1, max_genres=1) == {}
    assert build_genre_ground_truth(cfg, depth=1, max_genres=None)["u4"] == {"a1", "a2", "a3"}
