"""Item text construction for the embedding encoder.

The default format is title + categories concatenated. With GTE-large-en-v1.5
it outperformed a richer template (title | genres | description | features):
nDCG@10 0.2785 vs. 0.1793 (N=2000, seed=42).
"""


def prepare_book_text(item: dict) -> str:
    """Simple format: title + categories concatenated."""
    title = item.get("title", "").strip()
    cats = item.get("categories", [])
    if isinstance(cats, list):
        cats = " ".join(str(c) for c in cats if c)
    else:
        cats = str(cats).strip()

    text = f"{title} {cats}".strip()
    return text if text else ""


def prepare_album_text(item: dict) -> str:
    """Simple format: title + categories concatenated (domain-agnostic)."""
    title = item.get("title", "").strip()
    cats = item.get("categories", [])
    if isinstance(cats, list):
        cats = " ".join(str(c) for c in cats if c)
    else:
        cats = str(cats).strip()

    text = f"{title} {cats}".strip()
    return text if text else ""


def prepare_text_with_description(item: dict) -> str:
    """Rich format: title + categories + description.

    Categories come before the description so that the encoder's 512-token
    truncation cuts the description, never the categories.
    """
    title = item.get("title", "").strip()
    cats = item.get("categories", [])
    cats = " ".join(str(c) for c in cats if c) if isinstance(cats, list) else str(cats).strip()
    desc = item.get("description", "")
    desc = " ".join(str(d) for d in desc if d) if isinstance(desc, list) else str(desc or "")
    return " ".join(p for p in (title, cats, desc.strip()) if p)


def prepare_title_text(item: dict) -> str:
    """Title only: no categories and no description. An item without a title gives ``""``."""
    return (item.get("title") or "").strip()


# --text-format of the jsonl mode of the CLI
TEXT_FNS = {
    "simple": prepare_album_text,
    "desc": prepare_text_with_description,
    "title": prepare_title_text,
}
