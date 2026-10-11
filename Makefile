# Pipeline entry points. Each target is one documented step of docs/REPRODUCIBILITY.md.
# Run from the repository root with the virtualenv active.
PY ?= python
DOMAINS := music movies games
DOMAINS2 := movies2music movies2games  # movies as SOURCE (second source scenario)
MODEL := Alibaba-NLP/gte-large-en-v1.5
# domain:encoder:text of every run in results/granularity/ (music_gte_titlecat_legacy is added below)
GRID := music:gte:titlecat music:gte:desc music:gte:title music:bge-large:titlecat music:bge-m3:titlecat \
        music:e5-large-v2:titlecat music:minilm:titlecat music:minilm-l12:titlecat \
        music:nomic:titlecat music:qwen3:titlecat music:snowflake-arctic-embed-l:titlecat \
        music:uae-large-v1:titlecat \
        movies:gte:titlecat movies:bge-large:titlecat movies:qwen3:titlecat \
        games:gte:titlecat games:bge-large:titlecat games:qwen3:titlecat
# domain:encoder[:text] of the query-normalisation family: gte in every domain, every verified music
# encoder, and the description and title-only texts of music/gte (text defaults to titlecat)
QN_RUNS := music:gte music:gte:desc music:gte:title music:bge-large music:bge-m3 music:e5-large-v2 music:minilm music:nomic \
           music:snowflake-arctic-embed-l music:uae-large-v1 movies:gte games:gte
SEEDS := 43 44 45 46  # additional user splits of `multisplit` (the main split is seed 42)
MECH_ENCODERS := bge-large bge-m3 e5-large-v2 minilm minilm-l12 nomic qwen3 snowflake-arctic-embed-l uae-large-v1

.PHONY: install test lint data splits splits-new embeddings profiles whitening evaluate significance robustness figures grid stats hybrid mechanism query-norm lexical knn controls taxonomy lexical-overlap bridge splits-seeds multisplit reproduce-check paper-assets source2

install:
	pip install -e ".[dev]"

test:
	$(PY) -m pytest -q

lint:
	ruff check .

data:  ## 5-core filtering + ground truth per target domain (needs ACW_DATASETS_DIR)
	for d in $(DOMAINS); do $(PY) -m acw.data.preprocessing.cli preprocess --target-domain $$d --datasets-dir $$ACW_DATASETS_DIR; done

splits: data
	for d in $(DOMAINS); do $(PY) -m acw.data.preprocessing.cli build-canonical --target-domain $$d; done
	$(PY) -m acw.data.preprocessing.generate_canonical_split

embeddings:  ## item catalogs + source books (GPU recommended)
	$(PY) -m acw.embeddings.generate_embeddings.cli --model $(MODEL)

profiles:  ## H2 (mean-pooled) profiles; H3/H4 need an LLM server, see docs/REPRODUCIBILITY.md
	$(PY) -m acw.embeddings.generate_h2_profiles.cli --domain music --all-users

whitening:
	$(PY) -m acw.whitening --domain music

evaluate:
	for d in $(DOMAINS); do $(PY) -m acw.evaluation.evaluate --domain $$d --all --gt exact; done

significance:
	$(PY) -m acw.evaluation.significance_tests
	$(PY) -m acw.evaluation.significance_tests_aux_domains

robustness:
	$(PY) -m experiments.robustness.covariance_controls
	$(PY) -m experiments.robustness.offdomain_covariance
	$(PY) -m experiments.robustness.pca_vs_zca_and_collapse
	$(PY) -m experiments.robustness.popularity_control
	$(PY) -m experiments.robustness.raw_to_raw_alignment

figures:
	$(PY) -m experiments.figures.results_figures

# ── Current protocol: disjoint 70/10/20 splits, granularity grid, statistics, mechanism ──

splits-new:  ## data/splits/{domain}_{train,val,test}.json (needs `data` and the book embeddings)
	for d in $(DOMAINS); do $(PY) -m acw.data.splits --domain $$d; done

grid:  ## every domain x encoder run (about 1 h; runtimes and peak RSS in docs/REPRODUCIBILITY.md)
	for r in $(GRID); do set -- $$(echo $$r | tr ':' ' '); \
	  $(PY) -m experiments.granularity.run_corrections --domain $$1 --encoder $$2 --text $$3 || exit 1; done
	$(PY) -m experiments.granularity.run_corrections --domain music --encoder gte --text titlecat --legacy-population

stats:  ## paired tests and summary tables over the per-user results of `grid`
	$(PY) -m experiments.granularity.stats

hybrid:  ## content + lambda * popularity (gte, three domains), then the paired tests of the hybrids
	for d in $(DOMAINS); do $(PY) -m experiments.granularity.hybrid --domain $$d --encoder gte || exit 1; done
	$(PY) -m experiments.granularity.hybrid --stats --encoder gte

mechanism:  ## sections A-E (gte, three domains; needs query-norm for the centred-query column), A for the other music encoders, cross-encoder table
	for d in $(DOMAINS); do $(PY) -m experiments.granularity.mechanism --domain $$d --encoder gte || exit 1; done
	for e in $(MECH_ENCODERS); do $(PY) -m experiments.granularity.mechanism --domain music --encoder $$e --profile-only || exit 1; done
	$(PY) -m experiments.granularity.mechanism --music-encoders

query-norm:  ## query-side statistics (train users, books, target catalog) x center/ABTT/ZCA/Ledoit-Wolf, then the paired tests
	for r in $(QN_RUNS); do set -- $$(echo $$r | tr ':' ' '); \
	  $(PY) -m experiments.granularity.query_norm --domain $$1 --encoder $$2 --text $${3:-titlecat} || exit 1; done
	$(PY) -m experiments.granularity.query_norm --stats
	$(PY) -m experiments.granularity.query_norm --stats-val  # replication of the contrasts on the VAL users

lexical:  ## BM25 and IDF-overlap over the target titles (embedding-free), lexical + popularity, then the paired tests
	for d in $(DOMAINS); do $(PY) -m experiments.granularity.lexical --domain $$d || exit 1; done
	$(PY) -m experiments.granularity.lexical --domain music --text desc
	$(PY) -m experiments.granularity.lexical --domain music --text title
	$(PY) -m experiments.granularity.lexical --stats

knn:  ## profile-kNN over the train users (raw and centred profiles), kNN + content, then the paired tests
	for d in $(DOMAINS); do $(PY) -m experiments.granularity.knn --domain $$d --encoder gte || exit 1; done
	$(PY) -m experiments.granularity.knn --domain music --encoder gte --text title
	$(PY) -m experiments.granularity.knn --stats

controls:  ## shuffled-profile control (gte, three domains), then the paired tests
	for d in $(DOMAINS); do $(PY) -m experiments.granularity.controls --domain $$d --encoder gte || exit 1; done
	$(PY) -m experiments.granularity.controls --domain music --encoder gte --text title
	$(PY) -m experiments.granularity.controls --stats

bridge:  ## closed-form ridge bridge (book profile -> mean target embedding of the train users), lambda on VAL, then the paired tests
	for d in $(DOMAINS); do $(PY) -m experiments.granularity.bridge --domain $$d --encoder gte || exit 1; done
	$(PY) -m experiments.granularity.bridge --stats

lexical-overlap:  ## hit overlap of the query arms with the lexical scorers, reachability of the hits, semantic residual (after `query-norm`)
	for d in $(DOMAINS); do $(PY) -m experiments.granularity.lexical_overlap --domain $$d || exit 1; done
	$(PY) -m experiments.granularity.lexical_overlap --stats

splits-seeds:  ## data/splits/seed{S}/ for the additional splits
	for s in $(SEEDS); do for d in $(DOMAINS); do $(PY) -m acw.data.splits --domain $$d --seed $$s || exit 1; done; done

multisplit: splits-seeds  ## gte in three domains on each additional split (about 1 h per split), then the aggregation over the five splits
	for s in $(SEEDS); do \
	  for d in $(DOMAINS); do \
	    ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.run_corrections --domain $$d --encoder gte --text titlecat || exit 1; \
	    ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.query_norm --domain $$d --encoder gte || exit 1; \
	    ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.hybrid --domain $$d --encoder gte || exit 1; \
	    ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.knn --domain $$d --encoder gte || exit 1; \
	    ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.lexical --domain $$d || exit 1; \
	  done; \
	  ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.stats || exit 1; \
	  ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.query_norm --stats || exit 1; \
	  ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.query_norm --stats-val || exit 1; \
	  ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.hybrid --stats --encoder gte || exit 1; \
	  ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.knn --stats || exit 1; \
	  ACW_SPLIT_SEED=$$s $(PY) -m experiments.granularity.lexical --stats || exit 1; \
	done
	$(PY) -m experiments.granularity.multisplit

taxonomy:  ## category labels of the three target catalogs (results/granularity/taxonomy.json)
	$(PY) -m experiments.granularity.taxonomy

reproduce-check:  ## rerun music/gte/titlecat and compare with the committed summary
	$(PY) -m experiments.granularity.reproduce_check

source2:  ## movies as SOURCE (movies2music, movies2games): users, splits, grid, query-norm, kNN, lexical, bridge, stats
	$(PY) -m acw.data.movies_source
	for d in $(DOMAINS2); do $(PY) -m acw.data.splits --domain $$d || exit 1; done
	for d in $(DOMAINS2); do $(PY) -m experiments.granularity.run_corrections --domain $$d --encoder gte --text titlecat || exit 1; done
	for d in $(DOMAINS2); do $(PY) -m experiments.granularity.query_norm --domain $$d --encoder gte || exit 1; done
	$(PY) -m experiments.granularity.query_norm --stats --pairset source2
	$(PY) -m experiments.granularity.query_norm --stats-val --pairset source2
	for d in $(DOMAINS2); do $(PY) -m experiments.granularity.knn --domain $$d --encoder gte || exit 1; done
	$(PY) -m experiments.granularity.knn --stats --pairset source2
	for d in $(DOMAINS2); do $(PY) -m experiments.granularity.lexical --domain $$d || exit 1; done
	$(PY) -m experiments.granularity.lexical --stats --pairset source2
	for d in $(DOMAINS2); do $(PY) -m experiments.granularity.bridge --domain $$d --encoder gte || exit 1; done
	$(PY) -m experiments.granularity.bridge --stats --pairset source2
	$(PY) -m experiments.granularity.stats --runs $(addsuffix _gte_titlecat,$(DOMAINS2)) --tag _source2

PAPER_DIR ?= paper
paper-assets:  ## LaTeX tables and PDF figures of the article, from results/ only (PAPER_DIR=<article dir>)
	$(PY) -m experiments.figures.paper_tables --out $(PAPER_DIR)/tables
	$(PY) -m experiments.figures.granularity_figures --out $(PAPER_DIR)
	$(PY) -m experiments.figures.paper_numbers --paper $(PAPER_DIR)  # numbers.tex and NUMBERS.md; needs main.tex there
