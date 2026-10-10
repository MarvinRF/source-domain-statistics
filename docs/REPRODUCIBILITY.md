# Reproducibility Guide

How to regenerate every number behind the paper "Source-Domain Statistics, Not the Target Catalog:
What Geometric Corrections Do in Zero-Shot Text-Based Cross-Domain Recommendation" from the
raw Amazon Reviews 2023 files, under the **current protocol**: disjoint
70/10/20 user splits, a ladder of relevance granularities, per-user results,
paired statistics and mechanism diagnostics. Every command is run from the
repository root. The older genre-only protocol is kept at the end under
[Legacy protocol](#legacy-protocol).

Step map (each step is also a Makefile target):

| Step | Section | Makefile target |
|---|---|---|
| Environment, data, preprocessing | 1-2 | `install`, `data` |
| Embeddings | 3 and [EMBEDDINGS.md](EMBEDDINGS.md) | `embeddings` (item catalogs only) |
| User splits | 4 | `splits-new` |
| Granularity grid | 5 | `grid` |
| Paired statistics | 6 | `stats` |
| Mechanism diagnostics A-E | 7 | `mechanism` |
| Exact-vs-genre gate | 8 | - |
| Content plus popularity (hybrid) | 13 | `hybrid` |
| Query normalisation: which statistics, from which source, on the profile | 14 | `query-norm` |
| Lexical baselines (BM25, IDF overlap) | 15 | `lexical` |
| Profile-kNN collaborative baseline | 16 | `knn` |
| Shuffled-profile control; cold and warm items | 17 | `controls` |
| Title-only item text (no category labels) | 18 | rows of `grid`, `query-norm`, `lexical`, `knn`, `controls` |
| Rerun one configuration and diff it against the committed result | 11 | `reproduce-check` |
| Category taxonomy of the target catalogs | 7 | `taxonomy` |
| Tables, figures and number macros of the article | 12 | `paper-assets` |

## 1. Environment

```bash
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

All dependency versions are pinned in `pyproject.toml`. The provenance records
in the result JSONs were written with Python 3.14.4, numpy 2.4.4, scipy 1.17.1,
pandas 3.0.3, torch 2.11.0, transformers 5.8.0 and sentence-transformers 5.4.1
(each such JSON records the versions it ran with, see section 11). Python >= 3.11 is
required. A CUDA GPU is needed only for the embedding step; sections 4-8 run on
CPU.

Checks: `ruff check .`, `ruff format --check .` and `pytest` (CPU, no data needed).

## 2. Data

Download `Books`, `CDs_and_Vinyl`, `Movies_and_TV` and `Video_Games` review and
metadata files, plus the `Grocery_and_Gourmet_Food` metadata (off-domain
control), from the McAuley Lab Amazon Reviews 2023 release
(<https://amazon-reviews-2023.github.io>; not redistributed, see
[DATA.md](DATA.md)) and point the code to them:

```bash
export ACW_DATASETS_DIR=/path/to/amazon2023
for d in music movies games; do
  python -m acw.data.preprocessing.cli preprocess --target-domain $d --datasets-dir $ACW_DATASETS_DIR
done
```

Per domain this writes, under `data/raw/` (music) or `data/raw/<domain>/`
(movies, games): `books_filtered.jsonl` and `<target>_filtered.jsonl` (item
metadata), `user_book_history.json` (each user's chronological book history),
`ground_truth_real.json` (target items each user interacted with; used only to
define relevance), `overlapping_users.json` and `preprocessing_stats.json`.

The optional off-domain catalog used by the `zca_catalog_grocery` condition is
built with `python -m experiments.robustness.offdomain_catalog_build` from a
prefix of the grocery metadata saved as
`data/raw/grocery/meta_grocery_raw_sample.jsonl` (output:
`data/raw/grocery/grocery_filtered.jsonl`; see the script docstring).

## 3. Embeddings

Every encoder-and-text variant used by the grid is listed in
[EMBEDDINGS.md](EMBEDDINGS.md) with its file, model id, text format, shape,
sha256, producing command and verification status. Minimum for the
`gte`/`titlecat` runs (book and album catalogs, then movies and games):

```bash
python -m acw.embeddings.generate_embeddings.cli --model Alibaba-NLP/gte-large-en-v1.5 --batch-size 256
for d in movies games; do
  python -m acw.embeddings.generate_embeddings.cli --model Alibaba-NLP/gte-large-en-v1.5 --batch-size 256 \
      --jsonl-path data/raw/$d/${d}_filtered.jsonl --out-prefix $d --out-dir data/processed/$d
done
```

Other encoders, the description-text variant and the title-only variant (section 18) have
their own commands in EMBEDDINGS.md. GPU-produced embeddings can differ slightly across hardware
(GTE-large runs in fp16 on CUDA and fp32 on CPU), so the sha256 values there
identify the exact files behind the committed results; regenerated files are
expected to match them numerically but not necessarily bit for bit.

## 4. User splits

```bash
for d in music movies games; do python -m acw.data.splits --domain $d; done   # make splits-new
```

Eligible users have a non-empty exact ground truth (restricted to the indexed
catalog) and at least one embedded book. They are sorted, shuffled with
`acw.settings.SEED` (42) and cut into disjoint train/val/test lists (70/10/20).
Output: `data/splits/{domain}_{train,val,test}.json` and
`data/splits/{domain}_stats.json` (git-ignored, deterministic given the data).
Fitted quantities (profile ZCA, popularity, top category) use TRAIN users only.

**Other split seeds.** `python -m acw.data.splits --domain <d> --seed S` (`make splits-seeds`, seeds
43 to 46) writes the same files to `data/splits/seed{S}/`; the default seed 42 is the main split in
`data/splits/` itself and is unchanged. The experiments of section 5 and 13 to 17 read the split of the
environment variable `ACW_SPLIT_SEED` (default 42, `acw.settings.SPLIT_SEED`) and write their results to
`results/granularity/seed{S}/` (`acw.settings.GRANULARITY_DIR`); see section 19.

## 5. Granularity grid

One run per (domain, encoder, text):

```bash
python -m experiments.granularity.run_corrections --domain <d> --encoder <e> --text <t>
```

For the given encoder it builds H2 profiles (mean of the last 10 book
embeddings, L2-normalised), applies every correction in `acw.corrections`
(centering, ABTT, PCA, partial and full ZCA of the catalog, profile-side and
symmetric ZCA, ZCA fitted on books or groceries) plus the random, popularity
and top-category baselines, and scores val and test users under the exact
ground truth and the category-expansion ground truths `d3`/`d2`/`d1` on the
same users. Outputs: `results/granularity/<domain>_<encoder>_<text>.json`
(means, counts, category-depth statistics, input file list, provenance,
runtime, peak RSS) and the per-user `.csv.gz` (git-ignored; regenerate it with
the same command). `make grid` runs all of the following (about 52 minutes in
total, the sum of the recorded runtimes):

| Run | val / test users | train profiles | Granularities | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|---|---|
| `--domain music --encoder gte --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 118.0 | 2154 |
| `--domain music --encoder gte --text desc` | 2442 / 4884 | 17094 | exact, d2, d1 | 111.3 | 2138 |
| `--domain music --encoder gte --text title` | 2442 / 4884 | 17094 | exact, d2, d1 | 117.6 | 1947 |
| `--domain music --encoder bge-large --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 109.7 | 2138 |
| `--domain music --encoder bge-m3 --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 108.9 | 2140 |
| `--domain music --encoder e5-large-v2 --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 108.1 | 2144 |
| `--domain music --encoder minilm --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 85.4 | 1568 |
| `--domain music --encoder minilm-l12 --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 85.1 | 1568 |
| `--domain music --encoder nomic --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 98.9 | 1891 |
| `--domain music --encoder qwen3 --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 301.6 | 7418 |
| `--domain music --encoder snowflake-arctic-embed-l --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 109.9 | 2139 |
| `--domain music --encoder uae-large-v1 --text titlecat` | 2442 / 4884 | 17094 | exact, d2, d1 | 108.7 | 2139 |
| `--domain movies --encoder gte --text titlecat` | 6725 / 13458 | 47105 | exact, d2, d1 | 370.4 | 3225 |
| `--domain movies --encoder bge-large --text titlecat` | 6725 / 13458 | 47105 | exact, d2, d1 | 345.0 | 3188 |
| `--domain movies --encoder qwen3 --text titlecat` | 6725 / 13458 | 47105 | exact, d2, d1 | 763.3 | 8744 |
| `--domain games --encoder gte --text titlecat` | 978 / 1959 | 6857 | exact, d3, d2, d1 | 30.8 | 1731 |
| `--domain games --encoder bge-large --text titlecat` | 978 / 1959 | 6857 | exact, d3, d2, d1 | 28.2 | 1731 |
| `--domain games --encoder qwen3 --text titlecat` | 978 / 1959 | 6857 | exact, d3, d2, d1 | 88.4 | 6594 |
| `--domain music --encoder gte --text titlecat --legacy-population` | 2731 (val only) | 10920 | exact, d2 | 47.8 | 2058 |

Runtimes and peak RSS are the values stored in each result JSON (CPU only; the
`movies` and `qwen3` runs dominate). The `--text titlecat` label names the run,
not always the embedded text: the `bge-large` and screened encoders embed the
longer `text` field of the filtered jsonl (title, description, categories), and
the `bge-large` movies and games runs pair title + categories items with such
book embeddings; EMBEDDINGS.md gives the text per entry. A depth is skipped for a
domain when fewer than half of its catalog items have a category at that level
(hence no `d3` for music and movies). The movies and games `qwen3` runs are
flagged `provenance_unverified` in their JSON (see EMBEDDINGS.md). The
`--legacy-population` run evaluates the 2,731 canonical validation users of the
legacy protocol (users spanning more than six depth-2 categories excluded) as
a continuity check against the older numbers.

Popularity baselines never use information from the evaluated users: the
popularity list ranks catalog items by the number of TRAIN users whose exact
ground truth contains them (ties by item id; `acw.evaluation.pipeline.train_popularity`),
and the top-category list takes the category reached by most train users with
its items in that order. Only the `--legacy-population` run keeps the
older popularity, the metadata `rating_number` (ratings of all users, test users
included).

## 6. Paired statistics

```bash
python -m experiments.granularity.stats
```

Reads every per-user `results/granularity/<domain>_<encoder>_<text>.csv.gz` (the grid files
of section 5 only: not the `hybrid_*`, `query_norm_*`, `lexical_*`, `knn_*` or `controls_*` files of
sections 13-17) and, on the TEST split for nDCG@10 and Recall@100, writes:

- `stats_vs_raw.csv`: each condition vs `raw`. Every comparison of the repository (this
  section and sections 13-17) reports the same columns: `n`, the two means, `n_nonzero_diff` (users
  whose two values differ), `mean_diff` with a bootstrap 95% CI (2,000 resamples, seed 42), the
  PRIMARY test, a paired sign-flip (permutation) test on the mean difference (`p_signflip`, 10,000
  random sign vectors over the non-zero differences, seed 42, so the smallest possible p is 1e-4), the
  secondary Wilcoxon-Pratt test (`p_wilcoxon`) and the rank-biserial r (computed on the non-zero
  differences only). Holm is applied within each family = (domain, encoder, text, granularity, metric)
  over the corrections, WITHOUT `pca_catalog` (the PCA arm is an invalid control), to the sign-flip p
  (`p_holm_signflip`, primary) and to the Wilcoxon p (`p_holm_wilcoxon`); `p_holm` is the earlier
  Wilcoxon-Holm over the family WITH `pca_catalog`, kept unchanged for the tables that already use
  it. The baselines are reported unadjusted. `verified` is false for the encoders without a passed
  re-encoding audit (`qwen3`, `minilm-l12`, see EMBEDDINGS.md);
- `stats_vs_popularity.csv` and `stats_vs_top_category.csv`: `raw` and each correction against the
  constant popularity or top-category list, Holm over those conditions;
- `stats_tally.csv`: how many corrections are significantly better / worse / not different
  (Holm sign-flip p < 0.05, the sign of the mean difference) against each of the three references,
  per (granularity, metric) and in total (`gt = metric = all`), separately for verified and unverified
  encoders. `pca_catalog`, `raw` and the baselines are not counted;
- `stats_gap.csv` (granularity gap of the relative gain, with a paired bootstrap over users) and
  `summary_test.csv`, plus `stats_provenance.json` (code commit, versions, sha256 of the per-user
  files analysed).

It reads the per-user files, so run section 5 first. The columns that existed before the sign-flip
columns were added (`p_wilcoxon`, `p_holm`, CIs, `rank_biserial_r`) are reproduced unchanged.
All 18 non-legacy runs: 17 min 52 s wall clock, peak RSS 691 MB (CPU only); the sign-flip test dominates. The columns of the earlier tables (means, CIs, `p_wilcoxon`, `p_holm`, `rank_biserial_r`) were regenerated identical to the committed ones, also after the title-only run was added (the rows of the earlier runs are unchanged); `hybrid --stats` (section 13) takes 6 s.

## 7. Mechanism diagnostics

```bash
python -m experiments.granularity.mechanism --domain <music|movies|games> --encoder gte   # A-E
python -m experiments.granularity.mechanism --domain music --encoder <e> --profile-only   # A only
python -m experiments.granularity.mechanism --music-encoders                              # cross-encoder table
```

`make mechanism` runs all of these. Sections, on the TEST users of the
matching `results/granularity/<domain>_<encoder>_titlecat.csv.gz` (also used to
check that the rankings rebuilt here reproduce the per-user nDCG@10 of the
grid, tolerance 1e-4):

- **A** geometry of profiles (raw vs `zca_profile`) and catalog (raw vs
  `zca_catalog_a1.0` vs `center_catalog`): mean pairwise cosine, effective rank,
  variance share of PC1 and of the top 10 PCs.
- **B** retrieval diversity of the top-10 lists of raw, `zca_catalog_a1.0`,
  `zca_profile`, `zca_both_separate`, popularity and `center_profile` (the profile
  centred with the mean train profile, no whitening; its per-user nDCG@10 are checked against the
  `center_train_users` arm of `query_norm_<domain>_gte.csv.gz`, so section 14 must have been run
  first) over at most 3,000 sampled test users: catalog coverage, exposure Gini, mean list Jaccard, k-occurrence
  skewness, mean popularity percentile (rank of the number of train users who consumed the item).
- **C** category structure of the catalog (between/within variance ratio,
  same-category share of nearest neighbours).
- **D** Spearman correlation between raw-profile typicality and the nDCG@10
  gain of `zca_profile` over `raw`.
- **E** category concentration of the same top-10 lists (including `center_profile`) plus `top_category`
  (depth-2 category entropy; share in the 1 and the 5 most frequent train-user
  categories) and, per user, the Jaccard overlap with the constant
  `top_category` list and its Spearman correlation with the user's depth-2
  nDCG@10 (per condition and pooled; undefined, stored as null, when the
  overlap is constant). `pooled_personalised` covers the four original personalised conditions
  (raw, catalog ZCA, profile ZCA, both sides); `pooled_all` also contains the centred query.

Sections B-E need the catalog and exist only for the three `gte` runs; the
other music encoders have A (`--profile-only`). Outputs:
`results/mechanism/<domain>_<encoder>.json` and `music_encoders.json`. Adding the centred
query (`center_profile`) to B and E left every earlier value unchanged (the three files were
rerun and compared, except `E.jaccard_to_top_category.pooled_all`, which now includes it); a second
execution gave identical values.

```bash
python -m experiments.granularity.taxonomy      # make taxonomy
```

`taxonomy.py` describes the category labels of the evaluated catalog of each domain (items with a GTE
embedding) without embeddings or users: per position of the category path the number of distinct
labels, the five most frequent with their share of the items, and the share without a label at the
position, and the share of items whose first label is not the modal department (movies: 6,830 of
91,105 items). Output: `results/granularity/taxonomy.json`.

| Run | Sections | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|
| `--domain music --encoder gte` | A-E | 35.8 | 2101 |
| `--domain movies --encoder gte` | A-E | 98.4 | 3182 |
| `--domain games --encoder gte` | A-E | 10.0 | 1749 |
| `--domain music --encoder <e> --profile-only` (9 encoders) | A | 0.9 (minilm) to 36.8 (qwen3) | 766 (minilm) to 6416 (qwen3) |
| `--music-encoders` | cross-encoder table | not recorded | not recorded |

## 8. Exact-vs-genre gate

```bash
python -m experiments.main_results.gate_exact_gt                                        # music
python -m experiments.main_results.gate_exact_gt --emb-dir data/processed/desc --tag desc  # music, description text
python -m experiments.main_results.gate_exact_gt --domain movies
python -m experiments.main_results.gate_exact_gt --domain games
```

Compares raw, catalog-ZCA and profile-ZCA retrieval, with the Random,
popularity and top-genre baselines, under the exact and the genre ground truth
on the legacy canonical evaluation population, with a paired Wilcoxon test.
Popularity and top-genre are built from the legacy canonical TRAIN users (counts
of exact interactions, as in section 5), so they appear only where that split
exists (music); the movies and games gate files carry the Random baseline only.
Outputs: `results/gate_exact_gt_{canonical,desc}.json`,
`results/gate_exact_gt_{movies,games}_canonical.json`.

## 9. Which result file backs which analysis

| Analysis | Result file(s) | Produced by |
|---|---|---|
| Per-condition means by granularity (exact, d3, d2, d1; val and test), category-depth statistics | `results/granularity/<domain>_<encoder>_<text>.json` | `run_corrections` |
| Encoder sweep under the new protocol (11 music encoders) | `results/granularity/music_*_titlecat.json` | `run_corrections` |
| Description-text encoder | `results/granularity/music_gte_desc.json`, `results/gate_exact_gt_desc.json` | `run_corrections`, `gate_exact_gt` |
| Title-only text (section 18) | `results/granularity/music_gte_title.json`; `query_norm_music_gte_{desc,title}.json`, `lexical_music_title.json`, `knn_music_gte_title.json`, `controls_music_gte_title.json` (rows with `text = title` in the `stats_*` and `summary_*` tables) | `run_corrections --text title`, `query_norm`, `lexical`, `knn`, `controls` (each with `--text title`) |
| Legacy-population continuity check | `results/granularity/music_gte_titlecat_legacy.json` | `run_corrections --legacy-population` |
| Tests vs raw / vs popularity / vs top-category, granularity gap, summary table | `results/granularity/stats_vs_raw.csv`, `stats_vs_popularity.csv`, `stats_vs_top_category.csv`, `stats_gap.csv`, `summary_test.csv` | `stats` |
| Mechanism A-E (gte, three domains) | `results/mechanism/{music,movies,games}_gte.json` | `mechanism` |
| Mechanism A across music encoders and its correlation with the gain | `results/mechanism/music_<encoder>.json`, `music_encoders.json` | `mechanism` |
| Exact-vs-genre gate | `results/gate_exact_gt_*.json` | `gate_exact_gt` |
| Content plus popularity: lambda grid, validation curve, selected lambda, test means | `results/granularity/hybrid_<domain>_gte.json` | `hybrid` |
| Paired tests of the hybrids (vs popularity, vs the pure content, between hybrids) | `results/granularity/stats_hybrid.csv`, `stats_hybrid_provenance.json` | `hybrid --stats` |
| Query-side statistics, sources and arms; the contrasts of section 14; cold / warm rows | `results/granularity/query_norm_<domain>_<encoder>.json`, `stats_query_norm.csv`, `tally_query_norm.csv`, `summary_query_norm.csv` | `query_norm`, `query_norm --stats` |
| Lexical baselines and lexical + popularity | `results/granularity/lexical_<domain>.json`, `stats_lexical.csv`, `summary_lexical.csv` | `lexical`, `lexical --stats` |
| Profile-kNN and kNN + content | `results/granularity/knn_<domain>_gte.json`, `stats_knn.csv`, `summary_knn.csv` | `knn`, `knn --stats` |
| Shuffled-profile control | `results/granularity/controls_<domain>_gte.json`, `stats_controls.csv`, `summary_controls.csv` | `controls`, `controls --stats` |

Per-user `.csv.gz` files are not versioned (large); every number in them is
reproduced by rerunning the corresponding grid command (section 11 verifies
this for one configuration). Results without a `provenance` key were produced
before the provenance helper existed.

## 10. Determinism notes

- Seeds: `acw.settings.SEED = 42` drives the user splits, the random baseline,
  the bootstrap resamples in `stats`, and the user and pair sampling in
  `mechanism`. No other source of randomness is used.
- Fitting uses float64 accumulation over row blocks (`acw.corrections`); scoring
  and ranking use float32 matrix products and a stable sort of the top-k.
- Two executions of the same configuration on the same machine give a
  byte-identical summary (`means`) and per-user file (section 11). BLAS
  summation order can differ across machines, thread counts or library builds,
  which can in principle flip near-tied items in a top-k list. `OMP_NUM_THREADS=1` fixes the thread count.
- GPU embeddings (section 3) are the main cross-machine source of variation.
- Memory: the `qwen3` runs need up to about 9 GB; run heavy jobs one at a time.

## 11. Provenance and the reproducibility check

Every result JSON written by `run_corrections`, `mechanism`, `hybrid` and
`gate_exact_gt` (and `results/granularity/stats_provenance.json` and
`stats_hybrid_provenance.json` for `stats` and `hybrid --stats`)
carries a `provenance` key from `acw.provenance.provenance`: git commit, a
dirty flag (uncommitted changes under `src/`, `experiments/`, `pyproject.toml`),
Python and key package versions, a UTC timestamp, and the sha256 and size of
every input file (embeddings and id maps, metadata, ground truth, histories,
splits). Check an input against a result with `sha256sum`.

```bash
make reproduce-check     # python -m experiments.granularity.reproduce_check
```

reruns `music`/`gte`/`titlecat`, then requires every key of the new JSON except
`runtime_s`, `peak_rss_mb` and `provenance` to equal the committed one (the
`means` block is also compared as serialized text) and, when a per-user file
from an earlier run exists, its decompressed content to be identical. Exit
status 1 otherwise.

## 12. Tables and figures of the article

```bash
make paper-assets PAPER_DIR=/path/to/article   # paper_tables, granularity_figures, paper_numbers
```

`experiments/figures/paper_tables.py` writes one `tabular` per table to
`<PAPER_DIR>/tables/tab_*.tex` and `experiments/figures/granularity_figures.py`
writes `fig_granularity_ladder.pdf` and `fig_query_norm.pdf` to `<PAPER_DIR>`.
`experiments/figures/paper_numbers.py` writes `numbers.tex` (one `\\num<Name>` macro per number printed
in the prose of `<PAPER_DIR>/main.tex`, computed from the result files) and `NUMBERS.md` (macro, printed
value, section, source file and key); it fails if `main.tex` uses an undefined macro. All three read only `results/granularity/` and `results/mechanism/` (the two hybrid tables
`tab_hybrid` and `tab_hybrid_val` read `hybrid_*_gte.json` and `stats_hybrid.csv`; the tables of query
normalisation, lexical baselines, controls, cold and warm items, kNN, text formats and tallies read the
`summary_`, `stats_` and `tally_` files of sections 6 and 14-18, and Table 1 `taxonomy.json`; the table of
users with a hit also reads the per-user `*_gte_titlecat.csv.gz` of the three
domains, and Table 1 the user counts in `data/splits/*_stats.json` and
`data/raw/**/preprocessing_stats.json`). The provenance labels of the encoders
(verified or not, text embedded) are transcribed from `docs/EMBEDDINGS.md` in
`ENCODERS` of `paper_tables.py`.

## 13. Content plus popularity (hybrid)

```bash
python -m experiments.granularity.hybrid --domain <music|movies|games> --encoder gte
python -m experiments.granularity.hybrid --stats --encoder gte     # needs the three domains
```

`make hybrid` runs all of them. Question: does content add anything over the
train-based popularity list at item level? For a content condition C the score
of a (user, item) pair is `cosine_C(profile, item) + lambda * p(item)`, with
`p` the z-score over the catalog of `log(1 + n)` and `n` the number of TRAIN users whose
exact ground truth contains the item (`acw.evaluation.pipeline.train_popularity`).
Conditions: `raw`, `zca_profile` and `zca_catalog_a1.0` (the users, profiles and
fitted moments of section 5). Lambda grid: 0 (pure content), 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
1, 2, 5, 10 and infinity (the popularity ranking with exact ties broken by cosine; a
float64 score, so the cosine cannot overturn a count difference). The whole catalog is
ranked, top 100 per user, with one score matrix per batch of 512 users.

Lambda is chosen per (domain, C) on the VAL users by mean exact nDCG@10 (ties: the smaller
lambda); TEST metrics are reported for the chosen lambda only. Outputs:

- `results/granularity/hybrid_<domain>_gte.csv.gz` (git-ignored): per-user TEST metrics at
  the exact level and at depth 2 for random, popularity, top-category, the three pure
  conditions (`raw`, `zca_profile`, `zca_catalog_a1.0`) and the selected hybrids
  (`hybrid_<C>`);
- `results/granularity/hybrid_<domain>_gte.json`: grid, selected lambda, VAL curve (all
  metrics, exact level), TEST curve (diagnostic only, nothing is selected on it), TEST
  means, input hashes and provenance, runtime and peak RSS;
- `results/granularity/stats_hybrid.csv` and `stats_hybrid_provenance.json` (`--stats`): on
  TEST, exact nDCG@10 and Recall@100, each hybrid against popularity and against its own
  pure condition, and `hybrid_zca_profile` against `hybrid_raw` and
  `hybrid_zca_catalog_a1.0`; paired test as in section 6 (2,000 bootstrap resamples,
  seed 42; `n_nonzero_diff`, the primary sign-flip test `p_signflip` and its Holm-adjusted
  `p_holm_signflip`, the secondary Wilcoxon-Pratt `p_wilcoxon`), Holm within each (domain, metric)
  family of these 8 tests (`p_holm` is the Wilcoxon-Holm value, kept for the tables).

The run checks, before saving, that the pure conditions (lambda = 0) and the popularity
list reproduce the per-user nDCG@10 and Recall values of the grid run
`<domain>_gte_titlecat.csv.gz` (val and test, exact and depth 2) to within 1e-4, so section 5
must have been run first; the largest difference found is 5e-7 (the six-digit rounding of the
grid file). Two executions of the three domains and of `--stats` gave byte-identical
per-user files, summaries (except provenance, runtime and peak RSS) and statistics.

| Run | val / test users | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|
| `--domain music --encoder gte` | 2442 / 4884 | 109.5 | 2162 |
| `--domain movies --encoder gte` | 6725 / 13458 | 407.4 | 3419 |
| `--domain games --encoder gte` | 978 / 1959 | 21.1 | 1791 |

## 14. Query normalisation: which statistics, from which source, on the profile

```bash
python -m experiments.granularity.query_norm --domain <music|movies|games> --encoder <e> [--text <titlecat|desc|title>]
python -m experiments.granularity.query_norm --stats
```

`make query-norm` runs the twelve runs below and the tests (about 37 s for `--stats`). Question: which correction of the
PROFILE (the query), with statistics from which source, recovers the gain of profile-side ZCA?
The H2 profile of every user is transformed with the moments of a source S, fitted on the
`train_users` (the H2 profiles of the TRAIN users), the `books_catalog` (the source catalog) or
the `target_catalog`, and ranked by cosine against the RAW target catalog. Per source:
`center_S`, `abtt_S_D{1,3,10}`, `zca_S_a{0.25,0.5,0.75,1.0}` and `zcalw_S` (ZCA, alpha 1, with the
Ledoit-Wolf shrunk covariance of S: scikit-learn's intensity `delta`, eigenvalues replaced by
`(1 - delta) * lambda + delta * mean(lambda)`; the intensities are stored in the JSON). Also computed: `raw`, the catalog-side arms of
section 5 that act on the items with the catalog's own statistics (`center_catalog`,
`abtt_catalog_D{1,3,10}`, `zca_catalog_a{0.25,0.5,0.75,1.0}`) or the books' (`zca_catalog_books`),
`zca_catalog_train_users` (the catalog whitened with the train-profile covariance, profile raw),
`zca_both_separate`, and the random, popularity and top-category baselines. `pca_catalog`, the
off-domain grocery control and `zca_symmetric` are not part of the family.

`zca_train_users_a1.0` is the grid's `zca_profile`. Every arm that section 5 also computes and the
baselines are checked against its per-user file, val and test, exact and depth 2, to within 1e-4 (the
run aborts otherwise). VAL users (mean exact nDCG@10, ties to the first arm of the listed order) select the best arm of each source, the best query-side arm and the
best catalog-side arm; TEST metrics are reported for all of them under the exact ground truth,
depth-2 categories (`d2`) and the cold and warm subsets of the exact ground truth (an item is cold when no TRAIN user's ground truth contains it).
Runs: `gte` in the three domains and every music encoder marked verified in EMBEDDINGS.md
(`bge-large`, `bge-m3`, `e5-large-v2`, `minilm`, `nomic`, `snowflake-arctic-embed-l`,
`uae-large-v1`), plus `music`/`gte` with the description and the title-only item texts (`--text desc`,
`--text title`, section 18); the unverified `qwen3` and `minilm-l12` are not run (any registry entry can
be run by hand). The files of a text other than `titlecat` carry the suffix `_<text>`
(`query_norm_music_gte_title.json`); the grid check uses the grid file of the same text.

Outputs (`results/granularity/`):

- `query_norm_<domain>_<encoder>.csv.gz` (git-ignored): per-user TEST metrics;
- `query_norm_<domain>_<encoder>.json`: VAL and TEST means of every condition (nDCG@10, Recall@10/50/100,
  users, users with a hit at 10 and at 100) per granularity (`exact`, `d2`, `cold`, `warm`), the
  VAL-selected arms, the Ledoit-Wolf intensities, the grid check, provenance, runtime, peak RSS;
- `stats_query_norm.csv` (`--stats`): the paired tests of the contrasts fixed before any test number
  was seen, and of the VAL-selected arms, plus every arm fixed a priori against `raw`:
  (a) `zca_train_users_a1.0`, `center_train_users` and `zca_books_catalog_a1.0` each against
  `zca_catalog_a1.0`, and the best query-side arm against the best catalog-side arm (both chosen on
  VAL); (b) `zca_train_users_a1.0` against `center_train_users`; (c) `zca_both_separate` against
  `zca_train_users_a1.0`. Holm within each (domain, encoder, text, metric) family of the six contrasts and,
  separately, of the arms against `raw`; columns as in section 6 plus `text`. Three POST HOC contrasts,
  added after the first results were seen and forming their own Holm family (`family = post_hoc`):
  `zca_target_catalog_a1.0` and `center_target_catalog` against `raw`, and `zca_train_users_a1.0`
  against `zca_target_catalog_a1.0` (do the target-catalog statistics help on the query, and are the
  users' better?); the rows of the two earlier families are unchanged;
  A fourth Holm family, `family = matrix` (13 tests, defined after the third review round and therefore
  exploratory), covers the side x source matrix of ZCA (alpha 1): the six cells against `raw`
  (`matrix:{query,items}_{U,B,C}_vs_raw`; U = `train_users`, B = `books_catalog`, C = `target_catalog`;
  the item cells are `zca_catalog_train_users`, `zca_catalog_books`, `zca_catalog_a1.0`), the three
  matched-source side contrasts `matrix:side_{U,B,C}` (query minus items with the same source) and the
  source contrasts `matrix:source_{query,items}_{U,B}_vs_C`;
- `query_norm_<domain>_<encoder>_val.csv.gz` (git-ignored): the per-user VAL metrics of the run (exact level);
- `stats_query_norm_val.csv` (`--stats-val`, `make query-norm`): the same contrasts, arms against `raw`, post
  hoc and `matrix` tests on the VAL users (replication; the tests that use the VAL-selected arms are
  left out because the arms were chosen on those users). It exists for the runs that have a `_val` file
  (gte in the three domains and its description and title-only texts);
- `tally_query_norm.csv` (`--stats`): better / worse / not-different counts of those tests;
- `summary_query_norm.csv` (`--stats`): TEST means of every condition of every run.

| Run | val / test users | train profiles | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|---|
| `--domain music --encoder gte` | 2442 / 4884 | 17094 | 264.5 | 4680 |
| `--domain music --encoder gte --text desc` | 2442 / 4884 | 17094 | 259.9 | 4701 |
| `--domain music --encoder gte --text title` | 2442 / 4884 | 17094 | 265.9 | 4688 |
| `--domain music --encoder bge-large` | 2442 / 4884 | 17094 | 250.9 | 4680 |
| `--domain music --encoder bge-m3` | 2442 / 4884 | 17094 | 247.2 | 4680 |
| `--domain music --encoder e5-large-v2` | 2442 / 4884 | 17094 | 272.6 | 4692 |
| `--domain music --encoder minilm` | 2442 / 4884 | 17094 | 186.8 | 1565 |
| `--domain music --encoder nomic` | 2442 / 4884 | 17094 | 225.2 | 2462 |
| `--domain music --encoder snowflake-arctic-embed-l` | 2442 / 4884 | 17094 | 263.9 | 4692 |
| `--domain music --encoder uae-large-v1` | 2442 / 4884 | 17094 | 282.4 | 4688 |
| `--domain movies --encoder gte` | 6725 / 13458 | 47105 | 808.9 | 5293 |
| `--domain games --encoder gte` | 978 / 1959 | 6857 | 58.9 | 4293 |

About 57 minutes in total (CPU only); `--stats` takes about 32 s (peak RSS 666 MB). The largest per-user difference to the grid file over all runs is 5e-7 (the six-digit rounding of the grid file). Two executions of `--domain games --encoder gte` gave byte-identical per-user files and summaries (except provenance, runtime and peak RSS).

## 15. Lexical baselines

```bash
python -m experiments.granularity.lexical --domain <music|movies|games> [--text <titlecat|desc|title>]
python -m experiments.granularity.lexical --stats
```

`make lexical` runs the three domains, the `music` run with `--text title` and the tests. No embedding is
used. The query of a user is the
concatenation of the titles of the books of the H2 window (the last 10 history entries that are
embedded); an item is a document, either its title (`title`) or its title followed by its category
path (`titlecat`, the text that the title+categories encoders embed). Text is lower-cased and split
into runs of letters and digits (no stop list, no stemming). Scorers over the whole target catalog:
`bm25` (k1 = 1.2, b = 0.75, idf = ln(1 + (N - df + 0.5) / (df + 0.5)) with N and df from the catalog
documents, scipy sparse matrices, repeated query tokens count) and `idf_overlap` (sum of idf over the
distinct tokens shared by query and document). The four conditions are `bm25_title`, `bm25_titlecat`,
`idf_overlap_title`, `idf_overlap_titlecat`. Most items score zero, and ties are broken by ascending
item id, so no interaction log is used.

The hybrid `hybrid_<condition>` adds `lambda` times the train-popularity z-score of section 13 to the
lexical score divided, per user, by its maximum over the catalog (0 for a user without any matching
token). `lambda` comes from the grid of section 13 and is chosen per condition on the VAL users by mean exact
nDCG@10 (ties: the smaller lambda); `lambda = 0` is the pure lexical ranking and `lambda = inf` ranks by
popularity with ties broken by the lexical score. As a check of the pipeline, `strict_idf_full_window` and `strict_idf_h2_window` score a stricter,
independently written IDF-overlap matcher: letters-only tokens of at least four characters, a short stop list,
idf = ln(N / (1 + df)) over the item titles, distinct shared tokens, ties broken by `1e-6 * log(1 + n_train)`.
Its query in `strict_idf_full_window` is the titles of ALL of the last 10 history entries (titles from the
domain's `books_filtered.jsonl`, an unknown book having none); `strict_idf_h2_window` uses the H2 window of
this module (only the embedded books). The independent implementation's test nDCG@10 (users with a hit)
was 0.00437 (116), 0.00629 (489) and 0.00676 (84) in music, movies and games.

Outputs: `lexical_<domain>.csv.gz` (git-ignored, per-user TEST metrics), `lexical_<domain>.json` (lambda
grid, VAL curve, TEST curve for diagnosis only, selections, VAL and TEST means with the same
granularities as section 14, provenance), and with `--stats`: `stats_lexical.csv` (the lexical
conditions against `raw`, `center_train_users` and `zca_train_users_a1.0` of section 14 and against
popularity; the lexical hybrids against popularity, their pure lexical condition and the hybrids of
section 13; Holm per (domain, metric, family)) and `summary_lexical.csv`.

| Run | val / test users | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|
| `--domain music` | 2442 / 4884 | 156.8 | 1172 |
| `--domain movies` | 6725 / 13458 | 539.7 | 2325 |
| `--domain games` | 978 / 1959 | 25.8 | 529 |
| `--domain music --text title` | 2442 / 4884 | 162.6 | 1172 |

The lexical scores do not depend on the embedding text, so `--text` only names the run (suffix `_<text>`) and
pairs it, in `--stats`, with the embedding run of the same text (the hybrids of section 13 exist for
`titlecat` only, so the `title` rows carry no comparison with them). The `--domain music --text title` run is
identical to `--domain music` (per-user file, VAL and TEST means and selected lambdas), which is a second
execution of the same configuration. `--stats` takes about 30 s (peak RSS 672 MB). Two executions of `--domain games` gave byte-identical per-user files and summaries (except provenance, runtime and peak RSS). The pipeline check reproduces the independent implementation (`strict_idf_full_window`) to within 1% of its test nDCG@10 and to within one user of its hit counts: 0.00434 (115 users), 0.00625 (488) and 0.00671 (84) against 0.00437 (116), 0.00629 (489) and 0.00676 (84) in music, movies and games (the reference used arbitrary ties inside the top 10, this code breaks them by item id). With the H2 window of this module (`strict_idf_h2_window`) the values are 0.00434 (115), 0.00600 (479) and 0.00534 (72): in movies and games the reference query also used history books that have no embedding.

## 16. Profile-kNN collaborative baseline

```bash
python -m experiments.granularity.knn --domain <music|movies|games> --encoder gte [--text <titlecat|desc|title>]
python -m experiments.granularity.knn --stats
```

`make knn` runs the three domains, the `music` run with `--text title` and the tests. For a user, the `k` most similar TRAIN users are
found by cosine of their H2 profiles (`raw`: as they are; `center_train_users`: both sides centred with
the mean train profile and re-normalised); the score of a target item is the sum, over those
neighbours that interacted with it, of the similarity (so it uses the same train logs as the
popularity list). Items nobody in the neighbourhood interacted with score zero and ties are broken by ascending item id;
negative similarities are kept as they are (the number found among the 500 nearest is stored in the JSON as
`n_negative_neighbour_weights`). `k` in {10, 50, 100, 500} is chosen per variant on the VAL users by mean exact nDCG@10 (ties: the smaller
`k`). The hybrid adds `lambda` times the kNN score divided by its per-user maximum to the cosine of a content
condition (`raw` or `zca_train_users_a1.0`), for the neighbourhood of each variant at its selected `k`;
`lambda` comes from the grid of section 13 and is chosen on VAL per (variant, content). The pure content
conditions (`lambda = 0`) are checked against the grid files to within 1e-4.

Outputs: `knn_<domain>_<encoder>.csv.gz` (git-ignored), `knn_<domain>_<encoder>.json` (VAL and TEST
curves over `k` and over `lambda`, selections, means, grid check, provenance), and with `--stats`:
`stats_knn.csv` (kNN against popularity, `raw`, `zca_train_users_a1.0` and `center_train_users`; the
hybrids against popularity, their kNN, their content condition and the popularity hybrids of
section 13) and `summary_knn.csv`.

| Run | val / test users | train profiles | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|---|
| `--domain music --encoder gte` | 2442 / 4884 | 17094 | 173.2 | 2229 |
| `--domain movies --encoder gte` | 6725 / 13458 | 47105 | 602.5 | 3819 |
| `--domain games --encoder gte` | 978 / 1959 | 6857 | 31.2 | 1895 |
| `--domain music --encoder gte --text title` | 2442 / 4884 | 17094 | 172.5 | 2231 |

`--stats` takes about 28 s (peak RSS 665 MB). Two executions of `--domain games` gave byte-identical per-user files and summaries (except provenance, runtime and peak RSS). The VAL-selected `k` is 500, the largest of the grid, in every domain, for both variants and for the title-only text, so the grid is truncated at its upper edge. No negative similarity entered a neighbourhood. The pure content conditions (`lambda = 0`) differ from the grid files by at most 5e-7 per user.

## 17. Controls: shuffled profiles, cold and warm items

```bash
python -m experiments.granularity.controls --domain <music|movies|games> --encoder gte [--text <titlecat|desc|title>]
python -m experiments.granularity.controls --stats
```

`make controls` runs the three domains, the `music` run with `--text title` and the tests.

**Shuffled profiles.** The H2 profiles of the TEST users (for BM25, their queries) are permuted among
the TEST users with a seeded derangement (`acw.settings.SEED`, no user keeps their own profile), each user
keeping their own relevant items; the fitted statistics are unchanged. Conditions: `raw`,
`center_train_users`, `zca_train_users_a1.0`, `zca_catalog_a1.0`, `bm25_title`, `bm25_titlecat`, each
real and as `shuffled_<condition>`, scored under the exact and the depth-2 ground truth and on the cold and warm subsets.
`stats_controls.csv` (`--stats`) tests every shuffled condition against its real one (Holm per (domain, metric) family of six).

| Run | test users | Runtime (s) | Peak RSS (MB) |
|---|---|---|---|
| `--domain music --encoder gte` | 4884 | 43.0 | 2231 |
| `--domain movies --encoder gte` | 13458 | 133.4 | 2666 |
| `--domain games --encoder gte` | 1959 | 11.9 | 1895 |
| `--domain music --encoder gte --text title` | 4884 | 43.2 | 2233 |

`--stats` takes about 5 s (peak RSS 632 MB). Two executions of `--domain games` gave byte-identical per-user files and summaries (except provenance, runtime and peak RSS). The real conditions and the baselines differ from the grid files by at most 5e-7 per user.

**Cold and warm items.** An item is cold when no TRAIN user's ground truth contains it (popularity zero),
warm otherwise. Every run of sections 14-16 (and the shuffled conditions above) also scores each condition
and each baseline with the relevant set restricted to the user's cold items (`gt = cold`) or warm items
(`gt = warm`), with the same ranked lists; users with an empty restricted set are left out of that
granularity, so the number of users with at least one cold relevant item is the `n` of the `cold` rows
(`<g>_n_users` in the summary tables). The constant lists are the same as at the exact level.

## 18. Title-only item text

In the `titlecat` runs the category path is part of every item text (and of every book text that builds
the profiles), and the format was chosen on a genre-level metric. The `title` text removes the category
labels from both sides: albums and books are embedded from the title alone (GTE-large, EMBEDDINGS.md lists
the command, shape, sha256 and verification). Registry key `("music", "gte", "title")`; the catalogs, id
maps, splits, users and train profiles are the same as in the `titlecat` runs (2442 / 4884 val / test
users, 17094 train profiles). What remains of the categories is the relevance definition (`d2`, `d1`) and
the `top_category` baseline, which do not depend on the text.

```bash
python -m experiments.granularity.run_corrections --domain music --encoder gte --text title
python -m experiments.granularity.query_norm --domain music --encoder gte --text title
python -m experiments.granularity.query_norm --domain music --encoder gte --text desc   # same family, description text
python -m experiments.granularity.lexical --domain music --text title
python -m experiments.granularity.controls --domain music --encoder gte --text title
python -m experiments.granularity.knn --domain music --encoder gte --text title
python -m experiments.granularity.stats
for s in query_norm lexical knn controls; do python -m experiments.granularity.$s --stats; done
```

`make grid`, `make query-norm`, `make lexical`, `make knn`, `make controls` and `make stats` include these
runs. Result files carry the suffix `_title` (`music_gte_title`, `query_norm_music_gte_title`,
`lexical_music_title`, `knn_music_gte_title`, `controls_music_gte_title`); the `stats_*` and `summary_*`
tables of sections 6 and 14-17 gain a `text` column (`titlecat` for the first runs, whose rows are
unchanged), and the Holm families are formed within each (run, metric). The `zca_catalog_grocery` arm is
skipped for the `title` and `desc` texts (the grocery catalog is embedded with `titlecat`), and the
`query_norm`, `knn` and `controls` runs check the arms they share with the grid file of the same text to
within 1e-4 (largest difference 5e-7). A second execution of the grid run (`python -m experiments.granularity.reproduce_check --text title`) gave identical means and an identical per-user file. The title-only `knn --stats` and `lexical --stats` rows compare with the
title-only `query_norm` run; the hybrids of section 13 exist for `titlecat` only. The comparison across
the three texts (raw, centred, ZCA of the profile with train, book and catalog statistics, ZCA of the
catalog, top-category and popularity) is in `summary_query_norm.csv` and `stats_query_norm.csv`.

## 19. Additional user splits and split-to-split variability

The main results use one user split (seed 42). To measure how much the key contrasts move with the split,
the training-free pipeline is rerun on four more splits (seeds 43 to 46) for `gte` in the three domains:

```bash
make multisplit      # = make splits-seeds, then for each seed S the commands below, then the aggregation
# for S in 43 44 45 46 and d in music movies games:
python -m acw.data.splits --domain $d --seed $S
ACW_SPLIT_SEED=$S python -m experiments.granularity.run_corrections --domain $d --encoder gte --text titlecat
ACW_SPLIT_SEED=$S python -m experiments.granularity.query_norm --domain $d --encoder gte
ACW_SPLIT_SEED=$S python -m experiments.granularity.hybrid --domain $d --encoder gte
ACW_SPLIT_SEED=$S python -m experiments.granularity.knn --domain $d --encoder gte
ACW_SPLIT_SEED=$S python -m experiments.granularity.lexical --domain $d
# then, per seed: stats, query_norm --stats, query_norm --stats-val, hybrid --stats --encoder gte, knn --stats, lexical --stats
python -m experiments.granularity.multisplit
```

Every split has its own train / validation / test users (disjoint, 70/10/20), so everything that is fitted on
the training users (the moments of the profiles, the popularity list, the top category, the neighbourhoods),
the validation choices (the lambda of the hybrids, the k of the kNN, the arms selected for contrast (a4)) and
the test users change with the seed; the embeddings, the catalogs, the histories and the ground truth do not.
`ACW_SPLIT_SEED=S` makes `acw.data.splits.load_split` read `data/splits/seed{S}/` and every script write to
`results/granularity/seed{S}/`; the `run_corrections` file of the split is the reference of the grid checks
(1e-4) of the other runs, as for the main split. The `controls` and `mechanism` steps and the other
encoders and texts are not rerun.

`experiments.granularity.multisplit` reads the paired tests of the five splits (the main one and the four
extra ones) from `stats_query_norm.csv`, `stats_hybrid.csv`, `stats_knn.csv`, `stats_lexical.csv` and
`stats_vs_popularity.csv` (GTE, title+categories, exact nDCG@10, the Holm-adjusted sign-flip p of the split's own
family) and writes, to `results/granularity/`:

- `multisplit_per_split.csv`: one row per (contrast, domain, seed): paired mean difference, interval,
  `n_nonzero_diff`, Holm p, verdict (better / worse / ns);
- `multisplit_summary.csv`: per (contrast, domain), the mean and the sample standard deviation of the
  paired mean difference over the splits, its minimum and maximum, and the number of splits where it is
  significantly better, worse or not significant;
- `multisplit_levels.csv`: exact nDCG@10 and users with a hit of the main conditions per split;
- `multisplit_provenance.json`.

The sign-flip tests use the same seeds and sample sizes as the main split. The contrast list is the
`SPECS` table of the script.

## 20. Lexical overlap and semantic residual

```bash
make lexical-overlap     # = the three domains below, then --stats; needs `make query-norm lexical`
python -m experiments.granularity.lexical_overlap --domain <music|movies|games>
python -m experiments.granularity.lexical_overlap --stats
```

For `gte` / `titlecat` and the TEST users of one domain the script ranks the catalog with the raw, the
centred (`center_train_users`) and the ZCA (`zca_train_users_a1.0`) query and with the IDF-overlap and BM25
scorers over the item titles and the stricter IDF-overlap matcher (`strict_idf_h2_window`) of section 15; the
arms and the baselines are checked against the grid file (1e-4). It then computes:

- **A. hit overlap** (JSON key `A_hit_overlap`): for each (arm, lexical scorer) the users with a hit
  (exact nDCG@10 > 0) under each, under both, the number of both-hit users expected by chance when the two hit
  sets are independent (`n_a * n_b / n`) and when they are independent inside strata of the number of relevant
  items of the user (1, 2, 3-4, 5-9, 10+), the Jaccard index and a hypergeometric p (independence only);
- **B. reachability** (`B_reachability`): an item is *reachable* for a user when its title shares at least one
  rare token with the titles of the books of the user's window (the query of the lexical scorers; tokens as in
  section 15). A token is rare when it occurs in at most 1% of the catalog titles (0.1% as a sensitivity
  check; keys `0.01` and `0.001`). For each ranking, the share of its hit pairs (a relevant item among a user's
  top 10) that are reachable, and the chance levels: the share of the relevant items that are reachable and the
  mean share of catalog items that are reachable;
- **C. semantic residual**: the relevant set of each user is split into reachable items and the *residual*
  items (no shared rare token). Every ranking is scored again with the same lists against the residual set
  (`gt = resid`) and the reachable set (`gt = reach`; `resid@0.001` and `reach@0.001` for the sensitivity
  threshold); users with an empty set are left out. Nothing is re-embedded.

Outputs (`results/granularity/`): `lexical_overlap_<domain>.csv.gz` (git-ignored, per-user TEST metrics for
`exact`, `resid`, `reach`), `lexical_overlap_<domain>.json` (parameters, tables A and B, means and users with a
hit under every `gt`, grid check, provenance, runtime and peak RSS), and, with `--stats`,
`stats_lexical_overlap.csv` (paired sign-flip tests on the residual and reachable sets; Holm per domain and
`gt`), `summary_lexical_overlap.csv` and `stats_lexical_overlap_provenance.json`.

## Legacy protocol

The first version of the study used a single genre-expanded ground truth
(depth-2 categories, users spanning more than six categories excluded), a
2,731-user canonical validation split and learned alignment baselines. The
commands below regenerate those artifacts; the current protocol above does not
need them except `--legacy-population`, `gate_exact_gt` and the `encoder_sweep`
scripts, which read the canonical populations.

```bash
# Canonical populations (data/processed/canonical_{test,val,train}_users*.json)
for d in music movies games; do python -m acw.data.preprocessing.cli build-canonical --target-domain $d; done
python -m acw.data.preprocessing.generate_canonical_split
# Profiles (H2 needs no LLM; H3/H4 need an Ollama or vLLM backend)
python -m acw.embeddings.generate_h2_profiles.cli --domain music --all-users
python -m acw.embeddings.profile_summarization.cli --heuristic h3 --all-users --model llama3.2:3b
python -m acw.embeddings.profile_summarization.cli --heuristic h4 --all-users --model llama3.2:3b
# Whitening and genre-GT evaluation of every registered method
python -m acw.whitening --domain music
python -m acw.evaluation.evaluate --domain music --all
# Learned alignment baselines (repeat with --heuristic h4; --input-zca-book adds the ZB pre-stage)
python -m acw.baselines.parametric_whitening --domain music --heuristic h3
python -m acw.baselines.nonlinear_whitening   --domain music --heuristic h3
python -m acw.baselines.moe_whitening         --domain music --heuristic h3
python -m acw.baselines.infomax_alignment     --domain music --heuristic h3
# Tests on the legacy per-user scores
python -m acw.evaluation.significance_tests
python -m acw.evaluation.significance_tests_aux_domains
python -m experiments.main_results.headline_significance
# Robustness checks (outputs in results/)
python -m experiments.robustness.covariance_controls
python -m experiments.robustness.offdomain_covariance
python -m experiments.robustness.pca_vs_zca_and_collapse
python -m experiments.robustness.popularity_control
python -m experiments.robustness.raw_to_raw_alignment
python -m experiments.robustness.truncated_whitening
python -m experiments.robustness.larger_llm_pilot --n 60
# Nine-backbone sweep (BGE-large, screened and full-pipeline backbones, spectral predictors)
python -m experiments.encoder_sweep.bge_revalidation
python -m experiments.encoder_sweep.bge_significance
python -m experiments.encoder_sweep.full_pipeline
python -m experiments.encoder_sweep.full_pipeline_round2
python -m experiments.encoder_sweep.spectral_predictors
```

Pitfalls of the legacy pipeline: the evaluation silently prefers
`canonical_val_users.json` over `canonical_test_users_<domain>.json` when the
former exists (build profiles for the population the evaluation will select);
`ACW_EMBEDDING_DIM` must match the backbone (1024 for GTE/BGE-large, 4096 for
Qwen3-Embedding-8B, 384 for MiniLM, 768 for Nomic); omitting
`--input-zca-book` silently trains the non-ZB variant of PW/NL-PW/MoE-PW; the
games random baseline (nDCG@10 = 0.5011) is far higher than for music or
movies, so only same-domain deltas are comparable. Model ids of the nine
backbones: `Alibaba-NLP/gte-large-en-v1.5`, `Qwen/Qwen3-Embedding-8B`,
`BAAI/bge-large-en-v1.5`, `Snowflake/snowflake-arctic-embed-l`,
`sentence-transformers/all-MiniLM-L6-v2`, `nomic-ai/nomic-embed-text-v1.5`
(`trust_remote_code=True`), `intfloat/e5-large-v2`, `BAAI/bge-m3`,
`WhereIsAI/UAE-Large-V1`.
