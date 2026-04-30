from dataclasses import replace
import pytest

from src.chunking import add_parent_metadata, assign_chunk_ids, split_text
from src.retrieval import FaissSearchIndex, HybridSearchIndex, ParentChildSearchIndex, QdrantCloudSearchIndex, QdrantSearchIndex, TfidfSearchIndex, _batched, _qdrant_search, build_search_index


def test_roadmap_query_retrieves_roadmap_chunk():
    chunks = assign_chunk_ids(
        split_text("The roadmap prioritizes offline mode for July beta.", "a.txt")
        + split_text("Lunch menu includes soup and salad.", "b.txt")
    )
    results = TfidfSearchIndex(chunks).search("roadmap beta", top_k=1)
    assert results
    assert results[0].chunk.source_name == "a.txt"


def test_top_k_does_not_exceed_chunk_count():
    chunks = assign_chunk_ids(split_text("roadmap beta", "a.txt"))
    results = TfidfSearchIndex(chunks).search("roadmap", top_k=10)
    assert len(results) <= 1


def test_empty_index_is_safe():
    assert TfidfSearchIndex([]).search("anything", top_k=5) == []


def test_build_search_index_falls_back_for_missing_sentence_transformer(monkeypatch):
    chunks = assign_chunk_ids(split_text("roadmap beta", "a.txt"))

    def fail(_chunks):
        raise RuntimeError("model unavailable")

    monkeypatch.setattr("src.retrieval.SentenceTransformerSearchIndex", fail)
    index = build_search_index(chunks, "SentenceTransformer")

    assert isinstance(index, TfidfSearchIndex)


def test_hybrid_search_can_return_tfidf_hits_when_semantic_unavailable(monkeypatch):
    chunks = assign_chunk_ids(
        split_text("roadmap beta offline mode", "a.txt")
        + split_text("unrelated lunch menu", "b.txt")
    )
    monkeypatch.setattr("src.retrieval.SentenceTransformerSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))

    results = HybridSearchIndex(chunks).search("roadmap", top_k=1)

    assert results
    assert results[0].chunk.source_name == "a.txt"


def test_parent_child_expands_to_parent_text(monkeypatch):
    chunks = assign_chunk_ids(
        split_text("roadmap beta decision", "meeting.txt")
        + split_text("sync reliability risk", "meeting.txt")
    )
    chunks = [
        replace(chunk, metadata={**chunk.metadata, "parent_id": "parent_meeting"})
        for chunk in chunks
    ]
    monkeypatch.setattr("src.retrieval.SentenceTransformerSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))

    results = ParentChildSearchIndex(add_parent_metadata(chunks)).search("roadmap", top_k=1)

    assert results
    assert results[0].chunk.chunk_id == "parent_meeting"
    assert "sync reliability risk" in results[0].chunk.text


def test_faiss_backend_falls_back_when_unavailable(monkeypatch):
    chunks = assign_chunk_ids(split_text("roadmap beta", "a.txt"))
    monkeypatch.setattr("src.retrieval.FaissSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr("src.retrieval.SentenceTransformerSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))

    index = build_search_index(chunks, "FAISS")

    assert isinstance(index, TfidfSearchIndex)


def test_faiss_hybrid_uses_tfidf_when_faiss_unavailable(monkeypatch):
    chunks = assign_chunk_ids(split_text("roadmap beta", "a.txt"))
    monkeypatch.setattr("src.retrieval.FaissSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))

    index = build_search_index(chunks, "FAISS hybrid")
    results = index.search("roadmap", top_k=1)

    assert results


def test_qdrant_backend_falls_back_when_unavailable(monkeypatch):
    chunks = assign_chunk_ids(split_text("roadmap beta", "a.txt"))
    monkeypatch.setattr("src.retrieval.QdrantSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr("src.retrieval.FaissSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))

    index = build_search_index(chunks, "Qdrant")

    assert isinstance(index, TfidfSearchIndex)


def test_qdrant_hybrid_uses_tfidf_when_qdrant_unavailable(monkeypatch):
    chunks = assign_chunk_ids(split_text("roadmap beta", "a.txt"))
    monkeypatch.setattr("src.retrieval.QdrantSearchIndex", lambda _chunks: (_ for _ in ()).throw(RuntimeError()))

    index = build_search_index(chunks, "Qdrant hybrid")
    results = index.search("roadmap", top_k=1)

    assert results


def test_qdrant_search_uses_query_points_when_search_is_unavailable():
    class Response:
        points = ["hit"]

    class Client:
        def query_points(self, **kwargs):
            self.kwargs = kwargs
            return Response()

    client = Client()
    hits = _qdrant_search(client, "chunks", [0.1, 0.2], 3)

    assert hits == ["hit"]
    assert client.kwargs["collection_name"] == "chunks"
    assert client.kwargs["query"] == [0.1, 0.2]
    assert client.kwargs["limit"] == 3


def test_batched_splits_large_upserts():
    assert list(_batched(list(range(7)), 3)) == [[0, 1, 2], [3, 4, 5], [6]]


def test_qdrant_cloud_backend_raises_when_unavailable(monkeypatch):
    chunks = assign_chunk_ids(split_text("roadmap beta", "a.txt"))
    monkeypatch.setattr("src.retrieval.QdrantCloudSearchIndex", lambda _chunks, **_kwargs: (_ for _ in ()).throw(RuntimeError()))

    with pytest.raises(RuntimeError):
        build_search_index(
            chunks,
            "Qdrant Cloud",
            qdrant_cloud_config={"url": "https://example.com", "api_key": "secret", "collection_name": "test"},
        )
