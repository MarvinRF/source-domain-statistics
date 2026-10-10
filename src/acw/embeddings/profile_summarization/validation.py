"""Heuristic quality filters that reject malformed LLM profile summaries."""

# Markers that indicate the model echoed prompt instructions instead of
# generating a profile — always invalid regardless of heuristic.
_INSTRUCTION_ECHO_MARKERS = [
    "write only",
    "bullet points",
    "no introduction",
    "must cover",
    "as an ai",
]

# Markers that indicate evocative/literary language — only invalid for
# concrete heuristics where factual vocabulary is required.
_EVOCATIVE_MARKERS = [
    "sonic landscape",
    "mist-shrouded",
    "tapestry",
    "ancient secrets",
    "worn leather",
    "worn parchment",
    "whisper through",
    "mist-like",
    "hauntingly beautiful",
]


def validate_summary(text: str, heuristic: str = "") -> bool:
    if not text:
        return False

    # Concrete heuristics produce shorter factual output — lower threshold
    min_len = 30 if heuristic in ("h3_concrete", "h4_concrete") else 60
    if len(text) < min_len:
        return False

    low = text.lower()

    # Always reject instruction echoes
    if any(marker in low for marker in _INSTRUCTION_ECHO_MARKERS):
        return False

    # Reject evocative language only for concrete heuristics
    if heuristic in ("h3_concrete", "h4_concrete"):
        evocative_hits = sum(1 for m in _EVOCATIVE_MARKERS if m in low)
        if evocative_hits >= 2:
            return False

    # Reject wall-of-text outputs (prompt leakage)
    if low.count("\n") > 6:
        return False

    return True
