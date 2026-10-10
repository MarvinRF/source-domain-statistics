from experiments.granularity.taxonomy import summarise


def test_summarise_counts_labels_and_ragged_paths():
    paths = [["CDs", "Pop", "Oldies"], ["CDs", "Pop"], ["CDs", "Rock"], ["Comedy", "Camp"]]
    s = summarise(paths)
    assert s["n_items"] == 4 and s["position0"]["modal_label"] == "CDs"
    assert s["position0"]["n_not_modal"] == 1
    p1, p2 = s["positions"]["1"], s["positions"]["2"]
    assert p1["n_labels"] == 3 and p1["top"][0] == ["Pop", 0.5]
    assert p2["share_without_label"] == 0.75 and p2["n_labels"] == 1
