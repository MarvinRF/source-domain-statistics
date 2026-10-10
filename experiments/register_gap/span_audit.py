"""
Extraction-quality audit of Set B (evocative spans) in the query-side
register-gap test.

Set B (n=5,138) is extracted automatically with spaCy (POS pattern
ADJ{1,2} NOUN), so its extraction noise must be quantified. This script
redraws a fixed random sample (n=200, seed=42) from the existing CSV
(`results/phase4/dla_term_level_similarities.csv`, produced by
`experiments/register_gap/query_side_terms.py`), attaches a qualitative
label from 5 categories to each span, and reports the category shares and
the A-vs-B similarity gap both for the whole sample and for the
genuinely evocative spans only.

Categories:
  E = genuine evocative/atmospheric language about music (matches the
      "evocative descriptive language LLMs generate" being tested)
  G = factual/denotative genre term, same register as Set A (explicit
      genre names, technical/production jargon); runs against the
      "concrete vs. evocative" contrast
  L = literary/narrative content about the SOURCE domain (books: settings,
      plot, characters) rather than a description of music; inherited from
      the book-review corpus, not "evocative language about the target
      catalog"
  O = off-topic (related neither to music nor to plausible book-review
      content, e.g. cooking, gardening, tools)
  N = extraction artefact (incoherent/truncated span or POS-tagging error)

Supports: the robustness of the query-side register gap to extraction noise
in the automatically extracted evocative spans.

Input:  results/phase4/dla_term_level_similarities.csv
Output: results/phase4/dla_setB_quality_audit_sample.csv

Usage:
    python -m experiments.register_gap.span_audit
"""

import csv
from collections import Counter, defaultdict

from acw.settings import ROOT  # noqa: E402

SRC_CSV = ROOT / "results" / "phase4" / "dla_term_level_similarities.csv"
OUT_CSV = ROOT / "results" / "phase4" / "dla_setB_quality_audit_sample.csv"

# (term, label) — fixed random sample (n=200, seed=42) of the full Set B
# (n=5,138), sorted by decreasing corpus frequency.
# Labelled by a single rater; inter-rater agreement was not assessed
# (see the module docstring).
CLASSIFIED = [
    ("rainy afternoon", "E"),
    ("nostalgic melodies", "E"),
    ("nostalgic warmth", "E"),
    ("cinematic grandeur", "E"),
    ("eclectic blend", "E"),
    ("introspective interludes", "E"),
    ("sweeping orchestral score", "E"),
    ("stormy night", "E"),
    ("emotional resonance", "E"),
    ("high stakes", "L"),
    ("dramatic tension", "E"),
    ("heavy metal", "G"),
    ("sensory world", "E"),
    ("ancient wisdom", "L"),
    ("immersive worlds", "L"),
    ("cozy nights", "E"),
    ("electronic rock", "G"),
    ("intricate plots", "L"),
    ("acoustic tones", "G"),
    ("bustling café", "L"),
    ("hidden treasures", "L"),
    ("dark fantasy", "G"),
    ("contemplative elegance", "E"),
    ("humorous anecdotes", "L"),
    ("infectious energy", "E"),
    ("complex characters", "L"),
    ("symphonic suite", "G"),
    ("contemplative expanses", "E"),
    ("quirky charm", "E"),
    ("unvarnished truth", "L"),
    ("fireside gathering", "L"),
    ("simple joys", "E"),
    ("urban grit", "E"),
    ("delicate piano", "E"),
    ("nostalgic wanderlust", "E"),
    ("starry nights", "E"),
    ("symphonic piece", "G"),
    ("introspective rhythms", "E"),
    ("hidden worlds", "L"),
    ("industrial undertones", "E"),
    ("historic depth", "L"),
    ("heartfelt emotional cadence", "E"),
    ("intricate narrative", "L"),
    ("whimsical playfulness", "E"),
    ("playful harmonies", "E"),
    ("expansive soundscapes", "E"),
    ("together elements", "N"),
    ("introspective warmth", "E"),
    ("mournful strings", "E"),
    ("serene natural landscapes", "E"),
    ("vintage artifacts", "L"),
    ("dreamlike atmosphere", "E"),
    ("methodical builds", "G"),
    ("masterful blend", "E"),
    ("nostalgic threads", "E"),
    ("epic soundtracks", "G"),
    ("inspirational cadence", "E"),
    ("introspective depths", "E"),
    ("nostalgic vibes", "E"),
    ("ethereal whispers", "E"),
    ("historic cityscape", "L"),
    ("visceral thrill", "L"),
    ("playful jazz", "G"),
    ("lyrical introspection", "E"),
    ("imaginative escapades", "L"),
    ("golden atmospheric texture", "E"),
    ("symphonic crescendo", "E"),
    ("nocturnal cityscapes", "E"),
    ("together fragments", "N"),
    ("eclectic creativity", "E"),
    ("faint hum", "E"),
    ("sound effects", "G"),
    ("cozy studio", "E"),
    ("natural wonder", "L"),
    ("rugged adventure", "L"),
    ("witty wordplay", "L"),
    ("philosophical dialogue", "L"),
    ("quiet porch", "L"),
    ("soft scratch", "E"),
    ("heartfelt acoustic ballads", "G"),
    ("playful experimentation", "E"),
    ("suspenseful melodies", "E"),
    ("relaxed momentum", "E"),
    ("classic blues", "G"),
    ("orchestral post", "N"),
    ("insightful narratives", "L"),
    ("instrumental narrative", "G"),
    ("sensory backdrop", "E"),
    ("complex web", "N"),
    ("homemade cooking", "O"),
    ("introspective wanderlust", "E"),
    ("lilting arpeggios", "E"),
    ("smoky barroom", "L"),
    ("distant shore", "L"),
    ("solo guitar", "G"),
    ("historical flair", "L"),
    ("vibrant creativity", "E"),
    ("soft clickety", "N"),
    ("cinematic energy", "E"),
    ("humorous asides", "L"),
    ("moonlit waltz", "E"),
    ("intricate emotional cadences", "E"),
    ("literary narrative", "G"),
    ("vintage circus", "L"),
    ("electronic tracks", "G"),
    ("suspenseful narrative", "L"),
    ("aural novel", "N"),
    ("deserted battlefield", "L"),
    ("choral music", "G"),
    ("like settings", "N"),
    ("experimental genres", "G"),
    ("meandering road", "L"),
    ("harmonious dance", "E"),
    ("dramatic momentum", "E"),
    ("methodical beats", "G"),
    ("vintage tapestry", "E"),
    ("earthy atmosphere", "E"),
    ("playful instrumentation", "G"),
    ("natural imagery", "L"),
    ("vintage music", "G"),
    ("quirky soundscapes", "E"),
    ("creative workspaces", "O"),
    ("expansive atmospheres", "E"),
    ("offbeat realm", "E"),
    ("adventurous wonder", "L"),
    ("- noir soundtracks", "G"),
    ("lone traveler", "L"),
    ("classic instrumentation", "G"),
    ("indie ballads", "G"),
    ("unapologetic conviction", "L"),
    ("organic spaces", "E"),
    ("vast natural landscapes", "L"),
    ("atmospheric swells", "E"),
    ("blended edges", "N"),
    ("curious blend", "E"),
    ("introspective wandering", "E"),
    ("moonlit italian piazza", "L"),
    ("gentle ocean", "L"),
    ("nostalgic documentaries", "O"),
    ("mythic realm", "L"),
    ("rhythmic energy", "E"),
    ("outdoor adventure", "L"),
    ("poetic narrative", "L"),
    ("organic threads", "N"),
    ("soulful moments", "E"),
    ("new layer", "N"),
    ("sophisticated suspense", "L"),
    ("aromatic instrumentation", "N"),
    ("hypnotic soundscapes", "E"),
    ("classic road", "N"),
    ("vintage interiors", "L"),
    ("vibrant blooms", "O"),
    ("acoustic strumming", "E"),
    ("human curiosity", "L"),
    ("legendary performers", "G"),
    ("cosmopolitan environments", "L"),
    ("introspective beauty", "E"),
    ("vivid pictures", "L"),
    ("exploratory pace", "E"),
    ("reflective introspection", "E"),
    ("structured guides", "O"),
    ("urban commute", "O"),
    ("strategic builds", "N"),
    ("introspective awe", "E"),
    ("dramatic texture", "E"),
    ("acoustic environments", "G"),
    ("soft leather", "O"),
    ("subdued intensity", "E"),
    ("spoken word", "G"),
    ("rustic interiors", "L"),
    ("good novel", "O"),
    ("nostalgic landscape", "E"),
    ("primal intensity", "E"),
    ("colorful illustrations", "O"),
    ("moonlit outdoor", "N"),
    ("deserted cathedral", "L"),
    ("bustling cafe", "L"),
    ("exotic breeze", "L"),
    ("moonlit winter", "E"),
    ("essential oils", "O"),
    ("culinary masterpiece", "O"),
    ("cosmic landscapes", "E"),
    ("confessional tones", "E"),
    ("imaginative escapism", "L"),
    ("gritty world", "L"),
    ("tender intimacy", "E"),
    ("unapologetic candor", "L"),
    ("intricate curiosity", "N"),
    ("navigational tools", "O"),
    ("colorful children", "N"),
    ("nuanced expression", "E"),
    ("vintage amplifier", "G"),
    ("dusty pages", "O"),
    ("poignant soundscapes", "E"),
    ("funky basslines", "G"),
    ("golden emotional cadence", "E"),
    ("eclectic momentum", "E"),
    ("intricate puzzles", "O"),
    ("subconscious blur", "E"),
    ("vast open ranges", "L"),
]


def draw_sample():
    import random

    rows = []
    with open(SRC_CSV) as f:
        for row in csv.DictReader(f):
            if row["group"] == "B":
                rows.append(row)
    random.seed(42)
    sample = random.sample(rows, 200)
    sample.sort(key=lambda r: -int(r["frequency_in_corpus"] or 0))
    return sample


def main():
    sample = draw_sample()
    assert len(sample) == len(CLASSIFIED) == 200

    for i, (row, (term, _label)) in enumerate(zip(sample, CLASSIFIED), 1):
        if row["term"] != term:
            raise AssertionError(
                f"Misalignment at row {i}: csv={row['term']!r} vs classified={term!r}. "
                "The random sample must be reproduced exactly (seed=42) for this "
                "classification to remain valid."
            )

    counts = Counter(label for _, label in CLASSIFIED)
    print(f"n = {len(CLASSIFIED)}")
    for k in ["E", "G", "L", "O", "N"]:
        print(f"  {k}: {counts[k]} ({100 * counts[k] / 200:.1f}%)")

    sims_by_label = defaultdict(list)
    for row, (_, label) in zip(sample, CLASSIFIED):
        sims_by_label[label].append(float(row["mean_similarity_to_albums"]))

    main_rows = list(csv.DictReader(open(SRC_CSV)))
    a_sims = [float(r["mean_similarity_to_albums"]) for r in main_rows if r["group"] == "A"]
    a_mean = sum(a_sims) / len(a_sims)
    print(f"\nSet A (n={len(a_sims)}) mean sim to albums: {a_mean:.4f}")

    full_sample_mean = sum(float(r["mean_similarity_to_albums"]) for r in sample) / len(sample)
    print(
        f"Sample of B (n=200) mean sim: {full_sample_mean:.4f} -> gap vs A: "
        f"{(a_mean - full_sample_mean) / a_mean * 100:.2f}%"
    )

    e_mean = sum(sims_by_label["E"]) / len(sims_by_label["E"])
    print(
        f"Sample of B restricted to E-only (n={len(sims_by_label['E'])}) mean sim: "
        f"{e_mean:.4f} -> gap vs A: {(a_mean - e_mean) / a_mean * 100:.2f}%"
    )

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["term", "freq", "sim", "label"])
        for row, (_term, label) in zip(sample, CLASSIFIED):
            w.writerow(
                [
                    row["term"],
                    row["frequency_in_corpus"],
                    row["mean_similarity_to_albums"],
                    label,
                ]
            )
    print(f"\nSaved: {OUT_CSV}")


if __name__ == "__main__":
    main()
