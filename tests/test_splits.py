import pytest

from acw.data.splits import DOMAINS, SPLIT_NAMES, SPLITS_DIR, load_split, split_users


def test_split_users_is_disjoint_and_covers_everything():
    users = [f"u{i}" for i in range(1000)]
    s = split_users(users)
    assert set(s) == set(SPLIT_NAMES)
    assert sorted(sum(s.values(), [])) == sorted(users)
    for a in SPLIT_NAMES:
        assert s[a] == sorted(s[a])
        for b in SPLIT_NAMES:
            assert a == b or not set(s[a]) & set(s[b])
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (700, 100, 200)


def test_split_users_is_deterministic_and_order_independent():
    users = [f"u{i}" for i in range(50)]
    assert split_users(users) == split_users(users[::-1])


@pytest.mark.parametrize("domain", DOMAINS)
def test_user_split_files_are_disjoint(domain):
    if not all((SPLITS_DIR / f"{domain}_{s}.json").exists() for s in SPLIT_NAMES):
        pytest.skip(f"needs data/splits/{domain}_*.json (python -m acw.data.splits)")
    s = {name: set(load_split(domain, name)) for name in SPLIT_NAMES}
    assert all(s.values())
    assert not (s["train"] & s["val"] or s["train"] & s["test"] or s["val"] & s["test"])
    n = sum(len(v) for v in s.values())
    assert abs(len(s["train"]) / n - 0.7) < 0.01 and abs(len(s["val"]) / n - 0.1) < 0.01


def test_seed_selects_the_split_and_its_directory():
    from acw.data.splits import SPLITS_DIR, split_path, splits_dir
    from acw.settings import SEED

    assert splits_dir(SEED) == SPLITS_DIR  # the main split keeps its place
    assert split_path("music", "test", SEED) == SPLITS_DIR / "music_test.json"
    assert split_path("music", "test", 43) == SPLITS_DIR / "seed43" / "music_test.json"
    users = [f"u{i}" for i in range(200)]
    a, b = split_users(users, 43), split_users(users, 44)
    assert a != b and a == split_users(users[::-1], 43)  # another seed, another split
    assert (len(a["train"]), len(a["val"]), len(a["test"])) == (140, 20, 40)
