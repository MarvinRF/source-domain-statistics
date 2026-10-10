"""
Reproducibility check: rerun one run_corrections configuration and compare it
with the committed result.

The committed results/granularity/{domain}_{encoder}_{text}.json (git HEAD) is
the reference. The configuration is rerun with experiments.granularity.run_corrections
(overwriting the local result) and every key of the new summary must equal the
reference except the volatile ones (runtime_s, peak_rss_mb, provenance); the
``means`` block is additionally compared as serialized text. When a per-user
CSV of an earlier run exists locally (git-ignored), its decompressed content is
compared as well. Exit status 1 on any difference.

Usage:
    python -m experiments.granularity.reproduce_check
    python -m experiments.granularity.reproduce_check --domain movies --encoder gte
"""

from __future__ import annotations

import argparse
import gzip
import json
import subprocess
import sys

from acw.settings import GRANULARITY_DIR, ROOT

OUT_DIR = GRANULARITY_DIR

VOLATILE = ("runtime_s", "peak_rss_mb", "provenance")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--domain", default="music")
    ap.add_argument("--encoder", default="gte")
    ap.add_argument("--text", default="titlecat")
    args = ap.parse_args()

    stem = f"{args.domain}_{args.encoder}_{args.text}"
    json_path, csv_path = OUT_DIR / f"{stem}.json", OUT_DIR / f"{stem}.csv.gz"
    rel = json_path.relative_to(ROOT).as_posix()
    ref = json.loads(
        subprocess.run(
            ["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout
    )
    old_csv = gzip.open(csv_path).read() if csv_path.exists() else None

    subprocess.run(
        [sys.executable, "-m", "experiments.granularity.run_corrections"]
        + ["--domain", args.domain, "--encoder", args.encoder, "--text", args.text],
        cwd=ROOT,
        check=True,
    )
    new = json.loads(json_path.read_text())

    differing = sorted(
        k for k in (ref.keys() | new.keys()) - set(VOLATILE) if ref.get(k) != new.get(k)
    )
    means_text_equal = json.dumps(ref["means"], indent=2) == json.dumps(new["means"], indent=2)
    csv_equal = None if old_csv is None else old_csv == gzip.open(csv_path).read()
    print(f"{stem}: keys differing from HEAD (excluding {VOLATILE}): {differing or 'none'}")
    print(f"{stem}: means identical as serialized text: {means_text_equal}")
    print(f"{stem}: per-user CSV identical to the local earlier run: {csv_equal}")
    if differing or not means_text_equal or csv_equal is False:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
