"""
experiments/register_gap/query_side_terms.py
============================================
Term-level evidence for the query-side register gap (the "domain lexical
advantage"): do concrete catalog-style terms sit closer to the album catalog
in embedding space than the evocative language that LLM-generated user
profiles actually use? Replaces hand-curated examples with a fully
reproducible measurement.

Methodology (fixed before execution):
  - Set A (concrete terms): the closed vocabulary of the H3_concrete
    production prompt (acw prompts module), 50 terms.
  - Set B (evocative language): ADJ{1,2} NOUN spans extracted with spaCy
    from the full, cleaned H3 profile corpus
    (run_tag=llama70b_20260626_h4fix, n=24,420), frequency >= 5, no upper
    bound.
  - Embedding: Alibaba-NLP/gte-large-en-v1.5, normalize_embeddings=True
    (same convention as src/acw/embeddings/compare_embeddings.py).
  - Metric: mean similarity of each term to ALL 67,986 albums (not to the
    centroid).
  - Test: Mann-Whitney U + permutation test (n_resamples=10,000, seed=42)
    + rank-biserial r + 95% bootstrap CI of the mean difference.

Supports: the term-level (query-side) evidence for the register gap between
user-profile language and catalog language.

Inputs: data/processed/user_profiles_h3__llama70b_20260626_h4fix.json,
        data/processed/album_embeddings.npy; requires the spaCy model
        en_core_web_sm.
Output: results/phase4/dla_term_level_similarities.csv (every term with its
        individual similarity, for full auditability; the script refuses to
        overwrite an existing file). Consumed by
        experiments/register_gap/span_bootstrap.py.

Usage:
    python -m experiments.register_gap.query_side_terms
"""

from __future__ import annotations

import json
import re

import numpy as np
import spacy
from scipy.stats import mannwhitneyu, permutation_test

from acw.embeddings.profile_summarization.gte_encoder import GTEEncoder  # noqa: E402
from acw.settings import ROOT  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
SEED = 42
MIN_FREQ = 5

# ─── Set A: closed vocabulary of prompts.py::_H3_CONCRETE_PROMPT ───────────
GENRES = [
    "ambient",
    "post-rock",
    "slowcore",
    "dark folk",
    "jazz",
    "acoustic folk",
    "indie folk",
    "chamber pop",
    "modern classical",
    "shoegaze",
    "drone",
    "lo-fi",
    "progressive rock",
    "krautrock",
    "experimental",
    "black metal",
    "doom metal",
    "singer-songwriter",
    "gothic rock",
    "minimal techno",
    "dark ambient",
    "idm",
    "noise",
]
ENERGY = ["low", "medium", "high"]
MOOD = [
    "melancholic",
    "introspective",
    "atmospheric",
    "contemplative",
    "haunting",
    "peaceful",
    "tense",
    "somber",
    "ethereal",
    "aggressive",
    "uplifting",
    "euphoric",
    "brooding",
    "serene",
    "intense",
    "nostalgic",
    "sparse",
]
CONTEXT = [
    "late-night listening",
    "solitary reflection",
    "focus work",
    "background ambience",
    "morning commute",
    "social gathering",
    "active engagement",
]
SET_A = sorted(set(GENRES + ENERGY + MOOD + CONTEXT))
assert len(SET_A) == 50, f"expected 50 terms in Set A, got {len(SET_A)}"


def extract_evocative_spans(nlp, texts: list[str]) -> dict[str, int]:
    """Extract ADJ{1,2} NOUN spans via POS tagging; return {span: frequency}."""
    freq: dict[str, int] = {}
    set_a_lower = {t.lower() for t in SET_A}
    for doc in nlp.pipe(texts, batch_size=256):
        tokens = [t for t in doc if not t.is_space]
        i = 0
        n = len(tokens)
        while i < n:
            if tokens[i].pos_ == "ADJ":
                j = i + 1
                if j < n and tokens[j].pos_ == "ADJ":
                    j += 1
                if j < n and tokens[j].pos_ == "NOUN":
                    span = " ".join(t.text for t in tokens[i : j + 1])
                    span_norm = re.sub(r"[^\w\s-]", "", span.lower()).strip()
                    if span_norm and span_norm not in set_a_lower:
                        freq[span_norm] = freq.get(span_norm, 0) + 1
                    i = j + 1
                    continue
            i += 1
    return freq


def main() -> None:
    print("Loading spaCy en_core_web_sm...")
    nlp = spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])

    print("Loading cleaned H3 corpus (run_tag=llama70b_20260626_h4fix)...")
    with open(PROCESSED / "user_profiles_h3__llama70b_20260626_h4fix.json") as f:
        h3_data = json.load(f)
    texts = [v for v in h3_data.values() if v is not None]
    print(f"  {len(texts):,} valid H3 profiles")

    print("Extracting ADJ{1,2} NOUN spans (may take a few minutes)...")
    freq = extract_evocative_spans(nlp, texts)
    set_b = sorted([span for span, c in freq.items() if c >= MIN_FREQ])
    print(f"  {len(freq):,} unique spans found; {len(set_b):,} with frequency >= {MIN_FREQ}")

    print("Loading GTE-large-en-v1.5 via GTEEncoder (deterministic RoPE init)...")
    model = GTEEncoder(device="cpu")

    print("Loading album catalog (raw)...")
    albums = np.load(PROCESSED / "album_embeddings.npy").astype(np.float32)
    print(f"  {albums.shape}")

    all_terms = [("A", t) for t in SET_A] + [("B", t) for t in set_b]
    texts_to_embed = [t for _, t in all_terms]
    print(f"Embedding {len(texts_to_embed):,} terms/phrases...")
    embs = model.encode(
        texts_to_embed,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,
    ).astype(np.float32)

    print("Computing mean similarity of each term to all albums...")
    sims = embs @ albums.T  # (n_terms, n_albums)
    mean_sims = sims.mean(axis=1)

    out_path = ROOT / "results" / "phase4" / "dla_term_level_similarities.csv"
    if out_path.exists():
        raise FileExistsError(
            f"{out_path} already exists; refusing to overwrite it without an explicit decision."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("group,term,mean_similarity_to_albums,frequency_in_corpus\n")
        for (group, term), msim in zip(all_terms, mean_sims):
            freq_val = freq.get(term, "") if group == "B" else ""
            f.write(f'{group},"{term}",{msim:.6f},{freq_val}\n')
    print(f"Saved: {out_path}")

    a_vals = mean_sims[[i for i, (g, _) in enumerate(all_terms) if g == "A"]]
    b_vals = mean_sims[[i for i, (g, _) in enumerate(all_terms) if g == "B"]]

    print(f"\nSet A (n={len(a_vals)}): mean={a_vals.mean():.4f} std={a_vals.std():.4f}")
    print(f"Set B (n={len(b_vals)}): mean={b_vals.mean():.4f} std={b_vals.std():.4f}")
    rel_gap = (a_vals.mean() - b_vals.mean()) / a_vals.mean()
    print(f"Relative gap (A vs B): {rel_gap * 100:.1f}%")

    u_stat, p_mw = mannwhitneyu(a_vals, b_vals, alternative="two-sided")

    def statistic(x, y, axis=-1):
        return np.mean(x, axis=axis) - np.mean(y, axis=axis)

    rng = np.random.default_rng(SEED)
    res = permutation_test(
        (a_vals, b_vals),
        statistic,
        n_resamples=10_000,
        vectorized=True,
        alternative="two-sided",
        permutation_type="independent",
        random_state=rng,
    )
    n1, n2 = len(a_vals), len(b_vals)
    r_rb = 1 - (2 * u_stat) / (n1 * n2)

    rng2 = np.random.default_rng(SEED)
    boot_diffs = np.empty(10_000)
    for k in range(10_000):
        a_s = rng2.choice(a_vals, size=n1, replace=True)
        b_s = rng2.choice(b_vals, size=n2, replace=True)
        boot_diffs[k] = a_s.mean() - b_s.mean()
    ci_lo, ci_hi = np.percentile(boot_diffs, [2.5, 97.5])

    print(f"\nMann-Whitney U: U={u_stat:.1f}  p={p_mw:.3e}")
    print(f"Permutation test: p={res.pvalue:.4f}")
    print(f"Rank-biserial r: {r_rb:.3f}")
    print(f"Bootstrap 95% CI of the mean difference (A-B): [{ci_lo:.4f}, {ci_hi:.4f}]")


if __name__ == "__main__":
    main()
