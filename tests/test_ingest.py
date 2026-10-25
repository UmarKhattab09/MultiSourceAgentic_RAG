from ingest import chunk_text


def test_chunks_overlap_and_keep_metadata():
    doc = {"text": " ".join(f"w{i}" for i in range(100)), "metadata": {"source": "s.pdf", "page_number": 1}}
    chunks = chunk_text([doc], chunk_size=40, chunk_overlap=10)
    assert [c["text"].split()[0] for c in chunks] == ["w0", "w30", "w60", "w90"]
    assert chunks[1]["text"].split()[:10] == chunks[0]["text"].split()[-10:]
    assert all(c["metadata"]["source"] == "s.pdf" for c in chunks)
