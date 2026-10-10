# Embedding inventory

Every embedding file read by the encoder `REGISTRY` of
`experiments/granularity/run_corrections.py` (the granularity grid, and through
it `mechanism.py`) and by `experiments/main_results/gate_exact_gt.py`. Paths are
relative to `data/processed/`; row `i` of a `.npy` file is the item whose id maps
to `i` in the paired `*_id_to_idx*.json`. The loader (`_load_matrix`) checks
that the row count equals the id-map size and that the ids cover `0..n-1`, then
L2-normalises the rows.

The files were audited on 2026-10-08 (the title-only files on 2026-10-09, with
`experiments/tools/verify_item_embeddings.py`):

1. **Integrity** - shape, dtype, sha256 (also recorded in each result JSON under
   `provenance.inputs`), count of non-finite and zero rows (zero in every file),
   and id-map coverage (passes for every file).
2. **Re-encoding check** - for a seeded sample of 64 rows per file the item text
   was re-encoded with the documented model and text format, and the cosine to
   the stored row was computed (GPU, batch 16). "Verified" below means that
   check gave cosine >= 0.99999 for every sampled row. "Unverified" means it could
   not be run or did not reproduce the stored vectors; the reason is given.

GPU-produced embeddings may differ slightly across hardware: GTE-large runs in
fp16 on CUDA and fp32 on CPU, batching changes padding, and other encoders run
fp32 through sentence-transformers. The re-encoding check above agreed to
better than 1e-5 in cosine on the same GPU, but a different GPU or driver can
move vectors by more. The sha256 values identify the exact files behind the
committed results; regenerated files are expected to match numerically, not
bit for bit.

The registry flag `provenance_unverified` (set for the movies and games `qwen3`
entries) predates this audit and is stored in the grid JSONs; the status column
below is the audit outcome.

## Summary by registry entry

Registry key = (domain, encoder, text). The `text` of the key is the label of
the run, not always the text that was embedded: the "Text embedded" column is
what the files contain (determined by the re-encoding check).

| Registry key | Model | Item embeddings | Book embeddings | Text embedded | Status |
|---|---|---|---|---|---|
| music, gte, titlecat | `Alibaba-NLP/gte-large-en-v1.5` | `album_embeddings.npy` | `book_embeddings.npy` | title + categories | verified |
| movies / games, gte, titlecat | same | `movies/movies_embeddings_qwen.npy`, `games/games_embeddings_qwen.npy` | `book_embeddings.npy` | title + categories | verified |
| music, gte, desc | same | `desc/albums_embeddings_qwen.npy` | `desc/books_embeddings_qwen.npy` | title + categories + description | verified |
| music, gte, title | same | `title/albums_embeddings_qwen.npy` | `title/books_embeddings_qwen.npy` | title only (no categories; 4 albums without a title are embedded as the empty string) | verified |
| music / movies / games, gte, titlecat (condition `zca_catalog_grocery`) | same | (grocery catalog) `grocery/grocery_embeddings_qwen.npy` | - | title + categories | verified |
| music, bge-large, titlecat | `BAAI/bge-large-en-v1.5` | `bge_revalidation/albums_embeddings_bge.npy` | `bge_revalidation/books_embeddings_bge.npy` | albums and books: `text` field of `{albums,books}_filtered.jsonl` (title, description, categories, features; <= 1000 characters) | verified |
| movies / games, bge-large, titlecat | same | `bge_revalidation/{movies,games}_embeddings_bge.npy` | `bge_revalidation/books_embeddings_bge.npy` | movies and games: title + categories; **books: the `text` field** (a different template from the items) | verified |
| music, `bge-m3`, `e5-large-v2`, `minilm`, `nomic`, `snowflake-arctic-embed-l`, `uae-large-v1`, titlecat | see below | `backbone_screening/albums_embeddings_<tag>.npy` | `backbone_screening/books_embeddings_<tag>.npy` | `text` field of the filtered jsonl, with the encoder prefix where one is used | verified |
| music, minilm-l12, titlecat | `sentence-transformers/all-MiniLM-L12-v2` | `album_embeddings_checkpoint_all-MiniLM-L12-v2.npy` | `book_embeddings_checkpoint_all-MiniLM-L12-v2.npy` | not reproduced (see below) | **unverified** |
| music, qwen3, titlecat | `Qwen/Qwen3-Embedding-8B` | `album_embeddings_qwen3.npy` | `book_embeddings_qwen3.npy` | not re-encoded (see below) | **unverified** |
| movies / games, qwen3, titlecat | same | `{movies,games}/{movies,games}_embeddings_checkpoint__home_recsys_models_qwen3-embedding-8b.npy` | `book_embeddings_qwen3.npy` | not re-encoded | **unverified** |

`gate_exact_gt.py` reads `album_embeddings.npy` + `book_embeddings.npy` (music,
canonical), `desc/{albums,books}_embeddings_qwen.npy` (`--emb-dir
data/processed/desc`) and `{movies,games}/..._embeddings_qwen.npy` + `book_embeddings.npy`
(movies, games): the verified gte files above. All embeddings of the grid use
the book embeddings of the same encoder and text variant as the profile side.

## Commands that produced the files

Run from the repository root. Item texts are built by
`acw.embeddings.generate_embeddings.text_prep`: `title + " " + categories` (the
"simple" format) or, with `--text-format desc`, title + categories + description
or, with `--text-format title`, the title alone.
The encoder truncates at 512 tokens. For GTE-large the CLI uses
`acw.embeddings.profile_summarization.gte_encoder.GTEEncoder` (fp16 on CUDA);
the file `<prefix>_embeddings_checkpoint_<model>.npy` written next to every
output is byte-identical to the final output (checked for all gte files).

**GTE-large, music catalogs** (`album_embeddings.npy`, `book_embeddings.npy`; batch size and device from `embedding_stats.json`, which keeps only the last pass):

```bash
python -m acw.embeddings.generate_embeddings.cli --model Alibaba-NLP/gte-large-en-v1.5 --batch-size 256
```

**GTE-large, movies and games** (`--out-dir` inferred from where the files are; batch size 256, CUDA, from `{movies,games}_embedding_stats.json`):

```bash
for d in movies games; do
  python -m acw.embeddings.generate_embeddings.cli --model Alibaba-NLP/gte-large-en-v1.5 --batch-size 256 \
      --jsonl-path data/raw/$d/${d}_filtered.jsonl --out-prefix $d --out-dir data/processed/$d
done
```

**GTE-large, grocery catalog** (off-domain control; CPU, batch size 128, from `grocery_embedding_stats.json`; needs `python -m experiments.robustness.offdomain_catalog_build` first):

```bash
python -m acw.embeddings.generate_embeddings.cli --model Alibaba-NLP/gte-large-en-v1.5 --batch-size 128 --no-gpu \
    --jsonl-path data/raw/grocery/grocery_filtered.jsonl --out-prefix grocery --out-dir data/processed/grocery
```

**GTE-large, description text** (`desc/`). First copy the filtered catalogs adding the `description` field from the raw metadata (writes `data/raw/desc/{albums,books}_desc.jsonl`), then embed:

```bash
python -m acw.data.add_descriptions
python -m acw.embeddings.generate_embeddings.cli --model Alibaba-NLP/gte-large-en-v1.5 --batch-size 16 --jsonl-path data/raw/desc/{albums,books}_desc.jsonl --out-prefix {albums,books} --out-dir data/processed/desc --text-format desc
```

(`{albums,books}` is shell brace expansion shorthand: the embedding command is
run twice, once with `albums_desc.jsonl` / prefix `albums` and once with
`books_desc.jsonl` / prefix `books`; the stats files record batch size 16 on
CUDA, 2,036 s for albums and 7,024 s for books.)

**GTE-large, title-only text** (`title/`; the same filtered album and book catalogs as the canonical `titlecat` run, so the id maps are byte-identical to `album_id_to_idx.json` and `book_id_to_idx.json`). CUDA (RTX 3050 6 GB), batch size 64, no out-of-memory:

```bash
for d in albums books; do
  python -m acw.embeddings.generate_embeddings.cli --model Alibaba-NLP/gte-large-en-v1.5 --batch-size 64 \
      --jsonl-path data/raw/${d}_filtered.jsonl --out-prefix $d --out-dir data/processed/title --text-format title
done
python -m experiments.tools.verify_item_embeddings --emb title/albums_embeddings_qwen.npy --idx title/albums_id_to_idx_qwen.json --jsonl data/raw/albums_filtered.jsonl --text-format title
python -m experiments.tools.verify_item_embeddings --emb title/books_embeddings_qwen.npy --idx title/books_id_to_idx_qwen.json --jsonl data/raw/books_filtered.jsonl --text-format title
```

Albums: 67,986 items in 98.6 s (689 items/s; 1 min 48 s wall clock with the model load, peak RSS
3.4 GB). Books: 199,712 items in 468.7 s (426 items/s; 8 min 0 s wall clock, peak RSS 4.0 GB). 4
albums have no title (the books all do): they are kept in the catalog order and embedded as the
empty string (`n_sparse_metadata` = 4 in `title/albums_embedding_stats.json`; their four vectors agree to
1e-4, the fp16 padding noise). The re-encoding check of the second and third commands (64 rows, seed 42,
batch 16, GPU) gave a minimum cosine of 0.9999989 (albums) and 0.9999987 (books), medians 0.9999993 and
0.9999999: verified.

**BGE-large** (`bge_revalidation/`): `python -m experiments.encoder_sweep.bge_revalidation`
(sentence-transformers, CUDA, batch size 64, normalised; books and albums from
the `text` field of `data/raw/{books,albums}_filtered.jsonl`, movies and games from
`data/processed/{movies,games}/{movies,games}_texts.json`). The same script
writes the whitened catalogs and H2 profiles used by the older BGE analysis.

**Screened backbones** (`backbone_screening/`, `text` field of the filtered
jsonl, CUDA, normalised):

```bash
python -m experiments.encoder_sweep.full_pipeline                  # minilm (all-MiniLM-L6-v2), nomic (prefix "search_document: ")
python -m experiments.encoder_sweep.full_pipeline_round2 --only e5-large-v2   # prefix "passage: "
python -m experiments.encoder_sweep.full_pipeline_round2 --only bge-m3        # batch size 16
python -m experiments.encoder_sweep.full_pipeline_round2 --only snowflake-arctic-embed-l
python -m experiments.encoder_sweep.full_pipeline_round2 --only uae-large-v1
```

Model ids: `sentence-transformers/all-MiniLM-L6-v2`, `nomic-ai/nomic-embed-text-v1.5`,
`intfloat/e5-large-v2`, `BAAI/bge-m3`, `Snowflake/snowflake-arctic-embed-l`,
`WhereIsAI/UAE-Large-V1`.

## Unverified entries

- **movies and games `qwen3`.** Only the CLI checkpoint files survive
  (`..._checkpoint__home_recsys_models_qwen3-embedding-8b.npy`, written by the
  jsonl-mode CLI with `--model Qwen/Qwen3-Embedding-8B (local copy of the Hugging Face weights)`, inferred
  from the checkpoint name); the
  final `{movies,games}_embeddings_qwen.npy` names were later reused by the
  GTE runs. The files have the catalog row count, no zero or non-finite rows
  and match the id maps, but the exact command, batch size and text cannot be
  recovered, and the local weights are not available to re-encode.
- **music `qwen3`.** The album and book files are byte-identical to the CLI
  checkpoints `album_embeddings_checkpoint__home_recsys_models_qwen3-embedding-8b.npy`
  and `book_embeddings_checkpoint__home_recsys_models_qwen3-embedding-8b.npy`,
  i.e. written by the default-mode CLI (`python -m
  acw.embeddings.generate_embeddings.cli --model Qwen/Qwen3-Embedding-8B (local copy of the Hugging Face weights)`,
  inferred from the checkpoint name), which embeds the "simple" title +
  categories text. The weights
  (`Qwen/Qwen3-Embedding-8B`) are not available in this environment, so the
  re-encoding check could not be run. Row norms are 0.996-1.004 (half-precision
  output) rather than exactly 1.
- **music `minilm-l12`.** The files (modification time 14 May, written by the
  CLI's default model) do not reproduce from the current title + categories text:
  re-encoding gives a median cosine of 0.92 (albums, minimum 0.71) and 0.95
  (books, minimum 0.74). None of the other templates tried (title only, comma-
  or pipe-separated categories, `Titulo:` templates, the jsonl `text` field)
  matches either (best median 0.96), so the text used then is unknown. Row
  count and id order are consistent with the current id maps.

## The three music bge-large variants

Three different sets of BGE-large embeddings of the same music catalogs exist
(all `BAAI/bge-large-en-v1.5`, 1024 d, same id order):

| Variant | Files | Text embedded (re-encoding check) | Used by REGISTRY |
|---|---|---|---|
| `bge_revalidation` | `bge_revalidation/{albums,books}_embeddings_bge.npy` | `text` field of the filtered jsonl (cosine 1.0000 on 64 rows) | **yes** |
| default-mode checkpoint | `{album,book}_embeddings_checkpoint_BAAI_bge-large-en-v1.5.npy` | title + categories (cosine 1.0000) | no |
| older copy | `{album,book}_embeddings_bge-large.npy` | unknown: title + categories gives median 0.93 (albums) / 0.95 (books), the `text` field 0.77 / 0.79 | no |

The registry uses the `bge_revalidation` set because all four catalogs (books,
albums, movies, games) come from one script, `experiments/encoder_sweep/bge_revalidation.py`,
and that set is the one the BGE analyses of the paper read
(`experiments/encoder_sweep/bge_significance.py`,
`experiments/robustness/truncated_whitening.py`). The other two are different
vectors of the same model: row-wise cosine to the `bge_revalidation` albums
(5,000 sampled rows, matched by id) has mean 0.78 / 0.76 (older copy /
checkpoint), minimum 0.49 / 0.44 and maximum 0.97 / 0.97; for books the means are
0.78 / 0.77. Note that `experiments/encoder_sweep/spectral_predictors.py` points
its "BGE-large" entry at the older copy (`album_embeddings_bge-large.npy`), not
at the `bge_revalidation` set.

## Checksums

sha256 and shape of every file above; id maps follow.

### GTE-large-en-v1.5

| File | Rows x dim | sha256 |
|---|---|---|
| `album_embeddings.npy` | 67,986 x 1024 | `ec70534162d4edf1aac8b009bb89492b67dbe793383e3727a2f0083837b7b7cb` |
| `book_embeddings.npy` | 199,712 x 1024 | `a6032009a172bdd441579a484d731a81a64185a8e48885c3269e74e513510196` |
| `movies/movies_embeddings_qwen.npy` | 91,164 x 1024 | `2f51607961d67cf61a5f91e97fde284d5e74c3201a7e2dbf0d00871a9d9b0050` |
| `games/games_embeddings_qwen.npy` | 15,600 x 1024 | `5d96c9c096fb13fb5f671053e3a57a83449bb00c492602f3c1bfdd6a85b1918b` |
| `grocery/grocery_embeddings_qwen.npy` | 67,986 x 1024 | `880bc19f1d5e9267045fcc06038bdbf029be6c488291c76f0a14b115a570df96` |
| `desc/albums_embeddings_qwen.npy` | 67,986 x 1024 | `8b42f6b320af00865e3d409e21059f762122c587c5ab3dfa3e17d98fba451168` |
| `desc/books_embeddings_qwen.npy` | 199,712 x 1024 | `0b07bab6bb9a86e28e6589e1d48f867f06821ea0b26f2358ae19181619d24235` |
| `title/albums_embeddings_qwen.npy` | 67,986 x 1024 | `50c35af5ec0dcf79a44cd824831c99ba1cd3e52b84aac67e3c7c997ef5b1ff64` |
| `title/books_embeddings_qwen.npy` | 199,712 x 1024 | `85c360b2217e174cd2cb5177b7d5b609950b7539294a5dc3ad6f7944c1fb6c58` |

### BGE-large (`bge_revalidation/`, used) and the two unused music variants

| File | Rows x dim | sha256 |
|---|---|---|
| `bge_revalidation/albums_embeddings_bge.npy` | 67,986 x 1024 | `f9c8b7562ba202021ae93b86f5080c861a6455d6b41aa22f39a3d1b3cab2d522` |
| `bge_revalidation/books_embeddings_bge.npy` | 199,712 x 1024 | `fc1d16f9edef31257be0633be5701cb207f44a4c2f57eaad9c3cec00f04fd14a` |
| `bge_revalidation/movies_embeddings_bge.npy` | 91,164 x 1024 | `6ac5d415f54ce88f6cdd6c541449de552b9c9027f2e2b17f5887a9a6af4a5197` |
| `bge_revalidation/games_embeddings_bge.npy` | 15,600 x 1024 | `b10d50d2bb4f5cf852242b4df67e4636e83d674b6729101e3b8eec1394bd2de8` |
| `album_embeddings_checkpoint_BAAI_bge-large-en-v1.5.npy` (unused) | 67,986 x 1024 | `becbcf5a1591aa8c1f24756dab05cc673f6a0d9418c2692d120515732df23769` |
| `book_embeddings_checkpoint_BAAI_bge-large-en-v1.5.npy` (unused) | 199,712 x 1024 | `220cf7a907eb0a1789bcf8a72d183a2e32a64b9b0554cb1308a9e942424cb859` |
| `album_embeddings_bge-large.npy` (unused) | 67,986 x 1024 | `b8e05a7e360b788f18388a9e1b7c4f1ddc009f50698937a09290740cce74a884` |
| `book_embeddings_bge-large.npy` (unused) | 199,712 x 1024 | `702d3a3483f45d452c7a0265941738550b3c48c6b8742006ccddd9fdd41b156c` |

### Screened backbones (`backbone_screening/`)

| File | Rows x dim | sha256 |
|---|---|---|
| `backbone_screening/albums_embeddings_bge-m3.npy` | 67,986 x 1024 | `5397566485f1a20e00cfc219a8dde786f618197e8e514a9a9b2c9b75a8c47a4d` |
| `backbone_screening/books_embeddings_bge-m3.npy` | 199,712 x 1024 | `825f71be5ee6ff427504fdbb63118401ae486fca4f629a641ba1fc0758314feb` |
| `backbone_screening/albums_embeddings_e5-large-v2.npy` | 67,986 x 1024 | `0a3652ec175facbbc50547693c9d63870f57f0fa7b844c0df1b686e4394e7e76` |
| `backbone_screening/books_embeddings_e5-large-v2.npy` | 199,712 x 1024 | `798f3646ac36a2fd2e2e31c701da3ebc418eb2a883a28263c8d0d08f6f30fdcf` |
| `backbone_screening/albums_embeddings_minilm.npy` | 67,986 x 384 | `7b200ce9a28ebae0ef1b3964571ed38849ee3946ac6fa6f11ee9c7c4ff87970f` |
| `backbone_screening/books_embeddings_minilm.npy` | 199,712 x 384 | `b77c9cdbe40a862d85955241e7b856d651322fea01e52cb9dac7f7cd7966fc53` |
| `backbone_screening/albums_embeddings_nomic.npy` | 67,986 x 768 | `cac1e3b7281c41bb602d5e0231fb885ca8c35fabcbe68ddb130cbc2ecbbc0f65` |
| `backbone_screening/books_embeddings_nomic.npy` | 199,712 x 768 | `aa0348f01d576324e60c54fbb24f66b97a32cb922a9ba2a3a80c5e4aa535fdf5` |
| `backbone_screening/albums_embeddings_snowflake-arctic-embed-l.npy` | 67,986 x 1024 | `26e364daa6194c548b186bdc5a04bfa62946cd253d17b434aa49014cfbeb3137` |
| `backbone_screening/books_embeddings_snowflake-arctic-embed-l.npy` | 199,712 x 1024 | `17e7c7ae4a3e5e87697587131f6333618321eff8e5b76a7d3d0ff463b3bd0e5d` |
| `backbone_screening/albums_embeddings_uae-large-v1.npy` | 67,986 x 1024 | `b491b6405eaec30a2a17a8a5885369e496c3c131c74bed17f3bd206840e79cd8` |
| `backbone_screening/books_embeddings_uae-large-v1.npy` | 199,712 x 1024 | `8faa7a671d8a0ee69cae22497cec2b6aee8a01a2740226d0206dd230906377d5` |

### Unverified entries

| File | Rows x dim | sha256 |
|---|---|---|
| `album_embeddings_checkpoint_all-MiniLM-L12-v2.npy` | 67,986 x 384 | `d1052470d95f914e69fec5e0cea9961367a1da0d6a7a99d364a6bae0229ed866` |
| `book_embeddings_checkpoint_all-MiniLM-L12-v2.npy` | 199,712 x 384 | `b0714b5e6736ea4f4ac63f7ad737932424afb941c1e6a56afc978a94fd57df3a` |
| `album_embeddings_qwen3.npy` | 67,986 x 4096 | `99e6c160b52bdb9e2285078ca230d822e72a0b821bddf921d1edaade22ab6af2` |
| `book_embeddings_qwen3.npy` | 199,712 x 4096 | `0ab2c5ca9a16dc2d2654501a0750dbf15af6c4d254a4847c202ceec06d010018` |
| `movies/movies_embeddings_checkpoint__home_recsys_models_qwen3-embedding-8b.npy` | 91,164 x 4096 | `6df273efa8dd3faba03e3838cf797ff109c4453fc765d07fdcd2e520ecd96794` |
| `games/games_embeddings_checkpoint__home_recsys_models_qwen3-embedding-8b.npy` | 15,600 x 4096 | `ac0eb1b3c86f6f69c179986d560e77c961b1b980a47110820a2feef6f8bdeed7` |

### Id maps

The id maps of all variants of a catalog are byte-identical and list the items
in the order of the filtered jsonl (`data/raw/{albums,books}_filtered.jsonl`).

| Catalog | Files | sha256 |
|---|---|---|
| albums | `album_id_to_idx.json`, `desc/albums_id_to_idx_qwen.json`, `title/albums_id_to_idx_qwen.json`, `bge_revalidation/albums_id_to_idx_bge.json`, `backbone_screening/albums_id_to_idx_<tag>.json` | `30b057b3838b9385c9199dae4252870f1d5b00860d86e848c89164034a5facb1` |
| books | `book_id_to_idx.json`, `desc/books_id_to_idx_qwen.json`, `title/books_id_to_idx_qwen.json`, `bge_revalidation/books_id_to_idx_bge.json`, `backbone_screening/books_id_to_idx_<tag>.json` | `88fc9df9caadf43c7a83ccb3e86facbfe5eb6940bab2a7ced1d4c61e3906812a` |
| movies | `movies/movies_id_to_idx_qwen.json`, `bge_revalidation/movies_id_to_idx_bge.json` | `c7591659b8280655a5f8b2eaf4dceaff8b8427f4e4f6b4e7e164ee48685e13ce` |
| games | `games/games_id_to_idx_qwen.json`, `bge_revalidation/games_id_to_idx_bge.json` | `0a24b512b043e186f3d98ea7083121be6f21c2e35eb146202ee65172bb009f06` |
| grocery | `grocery/grocery_id_to_idx_qwen.json` | `760f6d20a37f7016e5da0f1dd1806029c60bfe4c94e9ef134b1d9d9808ee9931` |

(The `_qwen` suffix of the gte movies/games/grocery/desc/title files is the CLI's
generic output naming for `--jsonl-path` mode, not the Qwen3 model.)
