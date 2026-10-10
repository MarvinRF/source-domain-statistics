import hashlib

from acw.provenance import provenance, sha256_file


def test_sha256_streaming_matches_hashlib(tmp_path, monkeypatch):
    data = bytes(range(256)) * 1000
    f = tmp_path / "x.bin"
    f.write_bytes(data)
    monkeypatch.setattr("acw.provenance.CHUNK", 1000)  # several reads
    assert sha256_file(f) == hashlib.sha256(data).hexdigest()


def test_provenance_fields(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("abc")
    p = provenance([f, f])  # duplicates are hashed once
    assert list(p["inputs"].values()) == [
        {"sha256": hashlib.sha256(b"abc").hexdigest(), "bytes": 3}
    ]
    assert {"git_commit", "git_dirty", "python", "packages", "timestamp_utc"} <= set(p)
    assert set(p["packages"]) >= {
        "numpy",
        "scipy",
        "torch",
        "transformers",
        "sentence-transformers",
    }
