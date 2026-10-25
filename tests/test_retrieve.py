from retrieve import format_context, rrf, tokenize


def test_rrf_prefers_items_ranked_in_both_lists():
    a, b, c = ("a", {}), ("b", {}), ("c", {})
    assert rrf([[a, b], [b, c]])[0] == b


def test_rrf_keeps_every_item_once():
    a, b, c = ("a", {}), ("b", {}), ("c", {})
    assert sorted(doc for doc, _ in rrf([[a, b], [b, c]])) == ["a", "b", "c"]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("PerksPlus: $1,000!") == ["perksplus", "1", "000"]


def test_format_context_labels_sources():
    text = format_context([("chunk", {"source": "data/x.pdf", "page_number": 3}), ("md", {"source": "data/y.md"})])
    assert "[Source: x.pdf, page 3]" in text and "[Source: y.md]" in text
