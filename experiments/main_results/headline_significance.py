"""
Significance tests for the headline catalog-whitening result and for the
encoder-level (catalog-swap) robustness check, on the music domain with the
H2 user profiles.

Pairs (labels as written in the output CSV):
  B2:     H2-raw vs. H2 + album-fitted ZCA (the central raw-vs-whitened pair).
  B1/RC2: H2 + album-fitted ZCA vs. H2 + book-fitted ZCA (the whitening
          transform fitted on a different catalog; tests the encoder-level
          claim).

Each pair is tested with a paired Wilcoxon signed-rank test (Pratt zero
handling, Kerby rank-biserial r) and a paired permutation test on per-user
nDCG@10.

Inputs (results/statistical_tests/): per_user_music_retrains_H2+albuns-whitened.csv;
the H2-raw and H2-book-whitened per-user CSVs are generated via
acw.evaluation.per_user_metrics if missing (requires
data/processed/{user_profiles_h2,album_embeddings,book_embeddings}.npy).
Output: results/statistical_tests/significance_b1_b2.csv

Usage:
    python -m experiments.main_results.headline_significance
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import permutation_test

from acw.settings import ROOT  # noqa: E402

STATS_DIR = ROOT / "results" / "statistical_tests"
PROC = ROOT / "data" / "processed"


def _wilcoxon_r(a: np.ndarray, b: np.ndarray) -> tuple[float, float, float]:
    """Return (w_stat, p_wilcoxon, r).

    r = Kerby (2014) rank-biserial correlation, same formula as
    src/acw/evaluation/significance_tests.py: Pratt keeps zero differences,
    so the denominator uses the total n.
    """
    n = len(a)
    w_stat, p = stats.wilcoxon(a, b, zero_method="pratt")
    r = 1 - 2 * w_stat / (n * (n + 1) / 2)
    return w_stat, p, r


def _perm_p(a: np.ndarray, b: np.ndarray, n_resamples: int = 10_000, seed: int = 42) -> float:
    def stat(x, y, axis):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    res = permutation_test(
        (a, b), stat, n_resamples=n_resamples, random_state=seed, permutation_type="samples"
    )
    return res.pvalue


def run_test(label: str, csv_a: Path, csv_b: Path, col: str = "ndcg10") -> dict:
    da = pd.read_csv(csv_a)
    db = pd.read_csv(csv_b)
    m = da.merge(db, on="user_id", suffixes=("_a", "_b"))
    n = len(m)
    a, b = m[f"{col}_a"].values, m[f"{col}_b"].values
    _, p_wilcox, r = _wilcoxon_r(a, b)
    p_perm = _perm_p(a, b)
    return {
        "pair": label,
        "n": n,
        "mean_a": round(float(a.mean()), 4),
        "mean_b": round(float(b.mean()), 4),
        "delta": round(float(b.mean() - a.mean()), 4),
        "p_wilcoxon": p_wilcox,
        "p_permutation": p_perm,
        "r": round(r, 3),
    }


def main() -> None:
    # Generate the H2-raw CSV if missing
    raw_csv = STATS_DIR / "per_user_music_retrains_H2-raw.csv"
    if not raw_csv.exists():
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-m",
                "acw.evaluation.per_user_metrics",
                "--profiles",
                str(PROC / "user_profiles_h2.npy"),
                "--emb",
                str(PROC / "album_embeddings.npy"),
                "--label",
                "music_retrains_H2-raw",
                "--output",
                str(raw_csv),
            ],
            check=True,
        )

    # Generate the H2+book-whitened CSV if missing
    book_csv = STATS_DIR / "per_user_music_retrains_H2-book-whitened.csv"
    if not book_csv.exists():
        # Same path as the reported value (0.3736): ZCAWhitening fitted on
        # the books, applied to the albums, with L2 normalisation.
        from acw.whitening import ZCAWhitening

        books = np.load(str(PROC / "book_embeddings.npy")).astype(np.float32)
        albums = np.load(str(PROC / "album_embeddings.npy")).astype(np.float32)
        tmp = Path(tempfile.gettempdir()) / "album_embeddings_book_whitened.npy"
        np.save(tmp, ZCAWhitening().fit(books).transform(albums))
        import subprocess
        import sys

        subprocess.run(
            [
                sys.executable,
                "-m",
                "acw.evaluation.per_user_metrics",
                "--profiles",
                str(PROC / "user_profiles_h2.npy"),
                "--emb",
                str(tmp),
                "--label",
                "music_retrains_H2-book-whitened",
                "--output",
                str(book_csv),
            ],
            check=True,
        )

    wh_csv = STATS_DIR / "per_user_music_retrains_H2+albuns-whitened.csv"

    rows = [
        run_test("B2: H2-raw vs H2+album-ZCA", raw_csv, wh_csv),
        run_test("B1/RC2: H2+album-ZCA vs H2+book-ZCA", wh_csv, book_csv),
    ]
    df = pd.DataFrame(rows)
    out = STATS_DIR / "significance_b1_b2.csv"
    df.to_csv(out, index=False)
    print(df.to_string(index=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
