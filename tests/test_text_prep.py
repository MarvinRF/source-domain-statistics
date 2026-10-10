import json

from acw.embeddings.generate_embeddings.loader import load_jsonl
from acw.embeddings.generate_embeddings.text_prep import (
    TEXT_FNS,
    prepare_album_text,
    prepare_title_text,
)

ITEM = {"title": " Abbey Road ", "categories": ["Music", "Rock"], "description": ["A", "B"]}


def test_title_format_drops_categories_and_description():
    assert prepare_title_text(ITEM) == "Abbey Road"
    assert prepare_album_text(ITEM) == "Abbey Road Music Rock"
    assert TEXT_FNS["title"] is prepare_title_text


def test_title_format_of_a_missing_title_is_empty():
    assert prepare_title_text({"categories": ["Music"]}) == ""
    assert prepare_title_text({"title": None}) == ""


def _write(tmp_path, items):
    path = tmp_path / "items.jsonl"
    path.write_text("\n".join(json.dumps(i) for i in items))
    return path


def test_load_jsonl_keeps_untitled_items_in_order_only_with_keep_empty(tmp_path):
    path = _write(
        tmp_path,
        [
            {"parent_asin": "a", "title": "One", "categories": ["X"]},
            {"parent_asin": "b", "categories": ["X"]},
            {"parent_asin": "c", "title": "Three"},
        ],
    )
    asins, texts, empty = load_jsonl(path, prepare_title_text, "T", keep_empty=True)
    assert asins == ["a", "b", "c"] and texts == ["One", "", "Three"] and empty == ["b"]
    asins, texts, empty = load_jsonl(path, prepare_title_text, "T")
    assert asins == ["a", "c"] and empty == ["b"]  # default: dropped, as before
