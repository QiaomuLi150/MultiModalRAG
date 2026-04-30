from src.chunking import assign_chunk_ids, split_text


def test_empty_text_returns_no_chunks():
    assert split_text("", "empty.txt") == []


def test_long_text_splits_into_multiple_chunks():
    text = "roadmap " * 300
    chunks = split_text(text, "meeting.txt", chunk_size=200, overlap=40)
    assert len(chunks) > 1
    assert all(chunk.source_name == "meeting.txt" for chunk in chunks)


def test_chunk_ids_are_stable_sequence():
    chunks = assign_chunk_ids(split_text("alpha beta gamma", "notes.txt"))
    assert chunks[0].chunk_id == "chunk_001"

