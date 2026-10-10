"""LLM prompt templates and builders for the H3/H4 user-profile heuristics."""

from .config import MAX_BOOK_DESC, MAX_BOOKS_PROFILE

# ─────────────────────────────────────────────────────────────────────────────
# TEMPLATES
# ─────────────────────────────────────────────────────────────────────────────

_H3_PROMPT = """\
A user has positively rated the following books:

{book_list}

Your task is NOT to write a literary review.

Instead, synthesize the user's preferences into a semantic profile that could
align with MUSICAL atmosphere and listening experience.

Write ONE immersive paragraph (90–140 words) describing:

- emotional cadence
- atmospheric texture
- cinematic mood
- sonic analogies
- narrative intensity
- introspection vs momentum
- sensory aesthetics
- ideal listening environments associated with these preferences

The output should feel useful for matching:
books → albums → sonic atmospheres.

Focus on transferable experiential qualities rather than plot details.

The paragraph should sound natural, vivid, and semantically rich.

Write ONLY the paragraph.
No bullet points.
No labels.
No introduction.
"""

_H4_PROMPT = """\
A user has positively rated books with the following semantic profiles:

{entity_list}

Write ONE paragraph (90–140 words) describing this user's preferences in a way
that aligns with MUSICAL atmosphere and listening experience.

Anchor strongly to these normalized dimensions:

Atmosphere:
- melancholic
- epic
- distressing
- joyful
- contemplative
- dark
- hopeful
- neutral

Rhythm/Energy:
- low_introspective
- medium_fluid
- high_intense
- varied

Aesthetic:
- minimalist
- experimental
- classical
- baroque
- electronic
- acoustic
- orchestral

Context:
- nocturnal_solitary
- social_daytime
- festive_collective
- focus_work

The paragraph should:
- feel natural and immersive
- preserve the controlled vocabulary
- evoke sonic atmosphere and listening mood
- maximize semantic transferability into music retrieval space

Write ONLY the paragraph.
No labels.
No bullet points.
"""

_H3_CONCRETE_PROMPT = """\
A user has positively rated the following books:

{book_list}

Based on this reading history, generate a music preference profile using
SPECIFIC, FACTUAL MUSIC GENRE TERMINOLOGY.

Your response MUST follow this structure:

1. Genres: two to four specific genres from: ambient, post-rock, slowcore,
   dark folk, jazz, acoustic folk, indie folk, chamber pop, modern classical,
   shoegaze, drone, lo-fi, progressive rock, krautrock, experimental,
   black metal, doom metal, singer-songwriter, gothic rock, minimal techno,
   dark ambient, IDM, noise

2. Energy: exactly one: low / medium / high

3. Mood: two or three mood descriptors used in music metadata.
   Choose from: melancholic, introspective, atmospheric, contemplative,
   haunting, peaceful, tense, somber, ethereal, aggressive, uplifting,
   euphoric, brooding, serene, intense, nostalgic, sparse

4. Listening context: one or two:
   late-night listening / solitary reflection / focus work /
   background ambience / morning commute / social gathering /
   active engagement

RULES:
- Use ONLY concrete genre names and metadata vocabulary from the lists above
- Do NOT write literary metaphors, poetic imagery, or evocative prose
- Do NOT use phrases like "sonic landscape", "mist-shrouded", "tapestry",
  "ancient secrets", "worn leather", "whisper through the trees"
- Output ONE paragraph of 60 to 100 words

EXAMPLE OF CORRECT OUTPUT:
"This user would likely enjoy ambient, post-rock, and slowcore.
Energy: low. Mood: melancholic, introspective, and atmospheric.
Listening context: late-night listening and solitary reflection.
Artists like Stars of the Lid, Grouper, and Low align well with
this listener's contemplative, sparse aesthetic."

Write ONLY the paragraph. No labels. No bullet points.
"""

_H4_CONCRETE_PROMPT = """\
A user has positively rated books with the following semantic profiles:

{entity_list}

Based on these structured profiles, generate a music preference profile using
SPECIFIC, FACTUAL MUSIC GENRE TERMINOLOGY.

Translate entity vocabulary directly into music metadata vocabulary:
- atmosfera → mood descriptors (melancholic, introspective, somber, uplifting...)
- ritmo (low_introspective/medium_fluid/high_intense) → energy (low/medium/high)
- estetica (minimalist/baroque/experimental/acoustic...) → genre style
- contexto (nocturnal_solitary/social_daytime...) → listening context

Your response MUST follow this structure:
1. Genres: two to four specific genres from: ambient, post-rock, slowcore,
   dark folk, jazz, acoustic folk, indie folk, chamber pop, modern classical,
   shoegaze, drone, lo-fi, progressive rock, experimental, gothic rock,
   dark ambient, singer-songwriter, doom metal, krautrock, IDM, noise
2. Energy: exactly one: low / medium / high
3. Mood: two or three factual mood descriptors matching the atmosfera values
4. Listening context: one or two matching the contexto values

RULES:
- Translate entity vocabulary directly into music metadata vocabulary
- Do NOT write literary metaphors, poetic imagery, or evocative prose
- Output ONE paragraph of 60 to 100 words

Write ONLY the paragraph. No labels. No bullet points.
"""

_H3_MOVIES_PROMPT = """\
A user has positively rated the following books:

{book_list}

Your task is NOT to write a literary review.

Instead, synthesize the user's preferences into a semantic profile that could
align with CINEMATIC atmosphere and viewing experience.

Write ONE immersive paragraph (90–140 words) describing:

- emotional cadence
- visual and atmospheric texture
- narrative pacing and intensity
- genre-adjacent mood
- introspection vs momentum
- sensory aesthetics
- ideal viewing environments associated with these preferences

The output should feel useful for matching:
books → films → cinematic atmospheres.

Focus on transferable experiential qualities rather than plot details.

The paragraph should sound natural, vivid, and semantically rich.

Write ONLY the paragraph.
No bullet points.
No labels.
No introduction.
"""

_H4_MOVIES_PROMPT = """\
A user has positively rated books with the following semantic profiles:

{entity_list}

Write ONE paragraph (90–140 words) describing this user's preferences in a way
that aligns with CINEMATIC atmosphere and viewing experience.

Anchor strongly to these normalized dimensions:

Atmosphere:
- melancholic
- epic
- distressing
- joyful
- contemplative
- dark
- hopeful
- neutral

Pacing/Intensity:
- low_introspective
- medium_fluid
- high_intense
- varied

Aesthetic:
- minimalist
- experimental
- classical
- baroque
- electronic
- acoustic
- orchestral

Context:
- nocturnal_solitary
- social_daytime
- festive_collective
- focus_work

The paragraph should:
- feel natural and immersive
- preserve the controlled vocabulary
- evoke cinematic atmosphere and viewing mood
- maximize semantic transferability into film retrieval space

Write ONLY the paragraph.
No labels.
No bullet points.
"""

_H3_GAMES_PROMPT = """\
A user has positively rated the following books:

{book_list}

Your task is NOT to write a literary review.

Instead, synthesize the user's preferences into a semantic profile that could
align with INTERACTIVE/LUDIC atmosphere and gameplay experience.

Write ONE immersive paragraph (90–140 words) describing:

- emotional cadence
- atmospheric and world-building texture
- pacing and challenge intensity
- genre-adjacent mood
- introspection vs momentum
- sensory aesthetics
- ideal play environments associated with these preferences

The output should feel useful for matching:
books → games → interactive atmospheres.

Focus on transferable experiential qualities rather than plot details.

The paragraph should sound natural, vivid, and semantically rich.

Write ONLY the paragraph.
No bullet points.
No labels.
No introduction.
"""

_H4_GAMES_PROMPT = """\
A user has positively rated books with the following semantic profiles:

{entity_list}

Write ONE paragraph (90–140 words) describing this user's preferences in a way
that aligns with INTERACTIVE/LUDIC atmosphere and gameplay experience.

Anchor strongly to these normalized dimensions:

Atmosphere:
- melancholic
- epic
- distressing
- joyful
- contemplative
- dark
- hopeful
- neutral

Pacing/Challenge:
- low_introspective
- medium_fluid
- high_intense
- varied

Aesthetic:
- minimalist
- experimental
- classical
- baroque
- electronic
- acoustic
- orchestral

Context:
- nocturnal_solitary
- social_daytime
- festive_collective
- focus_work

The paragraph should:
- feel natural and immersive
- preserve the controlled vocabulary
- evoke interactive atmosphere and gameplay mood
- maximize semantic transferability into game retrieval space

Write ONLY the paragraph.
No labels.
No bullet points.
"""

# ─────────────────────────────────────────────────────────────────────────────
# BUILDERS
# ─────────────────────────────────────────────────────────────────────────────


def build_h3_prompt(book_texts: list[dict]) -> str:
    lines = []
    for i, book in enumerate(book_texts[:MAX_BOOKS_PROFILE], 1):
        title = book.get("title", "Unknown").strip()
        desc = str(book.get("description", "")).strip()[:MAX_BOOK_DESC]
        lines.append(f"{i}. {title}: {desc}" if desc else f"{i}. {title}")
    return _H3_PROMPT.format(book_list="\n".join(lines))


def build_h4_prompt(book_entities: list[dict]) -> str:
    lines = []
    for i, ent in enumerate(book_entities[:MAX_BOOKS_PROFILE], 1):
        atm = ent.get("atmosfera", "neutral")
        ritm = ent.get("ritmo", "medium_fluid")
        est = ent.get("estetica", "minimalist")
        ctx = ent.get("contexto", "nocturnal_solitary")
        lines.append(f"{i}. atmosphere={atm}, rhythm={ritm}, aesthetic={est}, context={ctx}")
    return _H4_PROMPT.format(entity_list="\n".join(lines))


def build_h3_concrete_prompt(book_texts: list[dict]) -> str:
    lines = []
    for i, book in enumerate(book_texts[:MAX_BOOKS_PROFILE], 1):
        title = book.get("title", "Unknown").strip()
        desc = str(book.get("description", "")).strip()[:MAX_BOOK_DESC]
        lines.append(f"{i}. {title}: {desc}" if desc else f"{i}. {title}")
    return _H3_CONCRETE_PROMPT.format(book_list="\n".join(lines))


def build_h4_concrete_prompt(book_entities: list[dict]) -> str:
    lines = []
    for i, ent in enumerate(book_entities[:MAX_BOOKS_PROFILE], 1):
        atm = ent.get("atmosfera", "neutral")
        ritm = ent.get("ritmo", "medium_fluid")
        est = ent.get("estetica", "minimalist")
        ctx = ent.get("contexto", "nocturnal_solitary")
        lines.append(f"{i}. atmosphere={atm}, rhythm={ritm}, aesthetic={est}, context={ctx}")
    return _H4_CONCRETE_PROMPT.format(entity_list="\n".join(lines))


def build_h3_movies_prompt(book_texts: list[dict]) -> str:
    lines = []
    for i, book in enumerate(book_texts[:MAX_BOOKS_PROFILE], 1):
        title = book.get("title", "Unknown").strip()
        desc = str(book.get("description", "")).strip()[:MAX_BOOK_DESC]
        lines.append(f"{i}. {title}: {desc}" if desc else f"{i}. {title}")
    return _H3_MOVIES_PROMPT.format(book_list="\n".join(lines))


def build_h4_movies_prompt(book_entities: list[dict]) -> str:
    lines = []
    for i, ent in enumerate(book_entities[:MAX_BOOKS_PROFILE], 1):
        atm = ent.get("atmosfera", "neutral")
        ritm = ent.get("ritmo", "medium_fluid")
        est = ent.get("estetica", "minimalist")
        ctx = ent.get("contexto", "nocturnal_solitary")
        lines.append(f"{i}. atmosphere={atm}, rhythm={ritm}, aesthetic={est}, context={ctx}")
    return _H4_MOVIES_PROMPT.format(entity_list="\n".join(lines))


def build_h3_games_prompt(book_texts: list[dict]) -> str:
    lines = []
    for i, book in enumerate(book_texts[:MAX_BOOKS_PROFILE], 1):
        title = book.get("title", "Unknown").strip()
        desc = str(book.get("description", "")).strip()[:MAX_BOOK_DESC]
        lines.append(f"{i}. {title}: {desc}" if desc else f"{i}. {title}")
    return _H3_GAMES_PROMPT.format(book_list="\n".join(lines))


def build_h4_games_prompt(book_entities: list[dict]) -> str:
    lines = []
    for i, ent in enumerate(book_entities[:MAX_BOOKS_PROFILE], 1):
        atm = ent.get("atmosfera", "neutral")
        ritm = ent.get("ritmo", "medium_fluid")
        est = ent.get("estetica", "minimalist")
        ctx = ent.get("contexto", "nocturnal_solitary")
        lines.append(f"{i}. atmosphere={atm}, rhythm={ritm}, aesthetic={est}, context={ctx}")
    return _H4_GAMES_PROMPT.format(entity_list="\n".join(lines))
