from __future__ import annotations

from dataclasses import dataclass
import uuid

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .chunking import DocumentChunk
from .encoders import TEXT_ENCODER_NAME


@dataclass
class SearchResult:
    chunk: DocumentChunk
    score: float


class TfidfSearchIndex:
    def __init__(self, chunks: list[DocumentChunk]):
        self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if self.chunks:
            self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
            self.matrix = self.vectorizer.fit_transform([chunk.text for chunk in self.chunks])
        self.encoder_name = TEXT_ENCODER_NAME

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip() or not self.chunks or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()
        ranked = scores.argsort()[::-1][: max(0, min(top_k, len(self.chunks)))]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked
            if scores[index] > 0
        ]


class SentenceTransformerSearchIndex:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: list[DocumentChunk], model_name: str | None = None):
        self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
        self.encoder_name = model_name or self.model_name
        self.model = None
        self.matrix = None
        if self.chunks:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(self.encoder_name)
            self.matrix = self.model.encode(
                [chunk.text for chunk in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip() or not self.chunks or self.model is None or self.matrix is None:
            return []
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        scores = np.dot(self.matrix, query_vector)
        ranked = scores.argsort()[::-1][: max(0, min(top_k, len(self.chunks)))]
        return [
            SearchResult(chunk=self.chunks[index], score=float(scores[index]))
            for index in ranked
        ]


class FaissSearchIndex:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: list[DocumentChunk], model_name: str | None = None):
        import faiss
        from sentence_transformers import SentenceTransformer

        self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
        self.encoder_name = f"faiss({model_name or self.model_name})"
        self.model = None
        self.index = None
        if self.chunks:
            self.model = SentenceTransformer(model_name or self.model_name)
            embeddings = self.model.encode(
                [chunk.text for chunk in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            self.index = faiss.IndexFlatIP(embeddings.shape[1])
            self.index.add(embeddings)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip() or not self.chunks or self.model is None or self.index is None:
            return []
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        ).astype("float32")
        scores, indices = self.index.search(query_vector, max(1, min(top_k, len(self.chunks))))
        return [
            SearchResult(chunk=self.chunks[int(index)], score=float(score))
            for score, index in zip(scores[0], indices[0])
            if int(index) >= 0
        ]


class QdrantSearchIndex:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    def __init__(self, chunks: list[DocumentChunk], model_name: str | None = None):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from sentence_transformers import SentenceTransformer

        self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
        self.encoder_name = f"qdrant(:memory:, {model_name or self.model_name})"
        self.collection_name = "multimodal_chunks"
        self.model = None
        self.client = None
        if self.chunks:
            self.model = SentenceTransformer(model_name or self.model_name)
            embeddings = self.model.encode(
                [chunk.text for chunk in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            self.client = QdrantClient(":memory:")
            self.client.recreate_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=embeddings.shape[1], distance=Distance.COSINE),
            )
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=index,
                        vector=embedding.tolist(),
                        payload={
                            "chunk_id": chunk.chunk_id,
                            "parent_id": chunk.metadata.get("parent_id"),
                            "source_name": chunk.source_name,
                            "modality": chunk.modality,
                            "timestamp": chunk.timestamp,
                            "page_or_frame": chunk.page_or_frame,
                            "text": chunk.text,
                        },
                    )
                    for index, (chunk, embedding) in enumerate(zip(self.chunks, embeddings))
                ],
            )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip() or not self.chunks or self.model is None or self.client is None:
            return []
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype("float32")
        hits = _qdrant_search(
            self.client,
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=max(1, min(top_k, len(self.chunks))),
        )
        return [
            SearchResult(chunk=self.chunks[int(hit.id)], score=float(hit.score))
            for hit in hits
        ]


class QdrantCloudSearchIndex:
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    upsert_batch_size = 64

    def __init__(
        self,
        chunks: list[DocumentChunk],
        url: str,
        api_key: str,
        collection_name: str = "multimodal_chunks",
        model_name: str | None = None,
    ):
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, PointStruct, VectorParams
        from sentence_transformers import SentenceTransformer

        if not url.strip() or not api_key.strip():
            raise ValueError("Qdrant Cloud URL and API key are required.")

        self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
        self.encoder_name = f"qdrant_cloud({model_name or self.model_name})"
        self.collection_name = collection_name.strip() or "multimodal_chunks"
        self.model = SentenceTransformer(model_name or self.model_name)
        self.client = QdrantClient(url=url.strip(), api_key=api_key.strip(), timeout=60)
        self.point_count = 0
        self.uploaded_count = 0
        self.before_count = self._safe_count_points()

        if self.chunks:
            embeddings = self.model.encode(
                [chunk.text for chunk in self.chunks],
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).astype("float32")
            self._ensure_collection(embeddings.shape[1])
            points = [
                PointStruct(
                    id=_point_id(chunk),
                    vector=embedding.tolist(),
                    payload=_payload_for_chunk(chunk),
                )
                for chunk, embedding in zip(self.chunks, embeddings)
            ]
            for batch in _batched(points, self.upsert_batch_size):
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=batch,
                    wait=True,
                )
            self.uploaded_count = len(self.chunks)
        self.point_count = self.count_points()

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip() or self.model is None:
            return []
        query_vector = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0].astype("float32")
        hits = _qdrant_search(
            self.client,
            collection_name=self.collection_name,
            query_vector=query_vector.tolist(),
            limit=max(1, top_k),
        )
        return [
            SearchResult(chunk=_chunk_from_payload(hit.payload or {}), score=float(hit.score))
            for hit in hits
        ]

    def _ensure_collection(self, vector_size: int) -> None:
        from qdrant_client.models import Distance, VectorParams

        existing = {collection.name for collection in self.client.get_collections().collections}
        if self.collection_name not in existing:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )

    def count_points(self) -> int:
        count = self.client.count(collection_name=self.collection_name, exact=True)
        return int(getattr(count, "count", 0))

    def _safe_count_points(self) -> int:
        try:
            return self.count_points()
        except Exception:
            return 0

    def status(self) -> dict:
        return {
            "collection_name": self.collection_name,
            "point_count": self.count_points(),
            "before_count": self.before_count,
            "uploaded_count": self.uploaded_count,
            "encoder_name": self.encoder_name,
        }


class HybridSearchIndex:
    def __init__(
        self,
        chunks: list[DocumentChunk],
        semantic_backend: str = "SentenceTransformer",
        qdrant_cloud_config: dict | None = None,
    ):
        self.tfidf = TfidfSearchIndex(chunks)
        try:
            if semantic_backend == "FAISS":
                self.semantic = FaissSearchIndex(chunks)
            elif semantic_backend == "Qdrant":
                self.semantic = QdrantSearchIndex(chunks)
            elif semantic_backend == "Qdrant Cloud":
                self.semantic = QdrantCloudSearchIndex(chunks, **(qdrant_cloud_config or {}))
            else:
                self.semantic = SentenceTransformerSearchIndex(chunks)
            self.encoder_name = f"hybrid({self.semantic.encoder_name}+{self.tfidf.encoder_name})"
        except Exception:
            if semantic_backend == "Qdrant Cloud":
                raise
            self.semantic = None
            self.encoder_name = f"hybrid_fallback({self.tfidf.encoder_name})"
        self.chunks = self.tfidf.chunks

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        fused: dict[str, tuple[DocumentChunk, float]] = {}
        if self.semantic is not None:
            for result in self.semantic.search(query, top_k=max(top_k * 4, top_k)):
                _merge_score(fused, result, weight=0.65)
        for result in self.tfidf.search(query, top_k=max(top_k * 4, top_k)):
            _merge_score(fused, result, weight=0.35)
        ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)
        return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked[:top_k]]

    def status(self) -> dict:
        if hasattr(self.semantic, "status"):
            return self.semantic.status()
        return {
            "collection_name": "",
            "point_count": len(self.chunks),
            "encoder_name": self.encoder_name,
        }


class ParentChildSearchIndex:
    def __init__(self, chunks: list[DocumentChunk], base_backend: str = "Hybrid"):
        self.chunks = [chunk for chunk in chunks if chunk.text.strip()]
        self.base_index = _build_search_index(self.chunks, base_backend, allow_parent_child=False)
        self.encoder_name = f"parent_child({getattr(self.base_index, 'encoder_name', base_backend)})"
        self.parents: dict[str, list[DocumentChunk]] = {}
        for chunk in self.chunks:
            parent_id = str(chunk.metadata.get("parent_id") or chunk.chunk_id)
            self.parents.setdefault(parent_id, []).append(chunk)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        child_hits = self.base_index.search(query, top_k=max(top_k * 4, top_k))
        parent_scores: dict[str, float] = {}
        for hit in child_hits:
            parent_id = str(hit.chunk.metadata.get("parent_id") or hit.chunk.chunk_id)
            parent_scores[parent_id] = max(parent_scores.get(parent_id, 0.0), hit.score)
        ranked_parent_ids = sorted(parent_scores, key=parent_scores.get, reverse=True)[:top_k]
        results: list[SearchResult] = []
        for parent_id in ranked_parent_ids:
            parent_chunks = self.parents[parent_id]
            representative = _combine_parent_chunks(parent_id, parent_chunks)
            results.append(SearchResult(chunk=representative, score=parent_scores[parent_id]))
        return results


def build_search_index(chunks: list[DocumentChunk], backend: str, qdrant_cloud_config: dict | None = None):
    return _build_search_index(
        chunks,
        backend,
        allow_parent_child=True,
        qdrant_cloud_config=qdrant_cloud_config,
    )


def _build_search_index(
    chunks: list[DocumentChunk],
    backend: str,
    allow_parent_child: bool,
    qdrant_cloud_config: dict | None = None,
):
    backend = backend.replace(" + step-back", "")
    if backend == "FAISS":
        try:
            return FaissSearchIndex(chunks)
        except Exception:
            try:
                return SentenceTransformerSearchIndex(chunks)
            except Exception:
                return TfidfSearchIndex(chunks)
    if backend == "FAISS hybrid":
        return HybridSearchIndex(chunks, semantic_backend="FAISS")
    if backend == "Qdrant":
        try:
            return QdrantSearchIndex(chunks)
        except Exception:
            try:
                return FaissSearchIndex(chunks)
            except Exception:
                return TfidfSearchIndex(chunks)
    if backend == "Qdrant hybrid":
        return HybridSearchIndex(chunks, semantic_backend="Qdrant")
    if backend == "Qdrant Cloud":
        return QdrantCloudSearchIndex(chunks, **(qdrant_cloud_config or {}))
    if backend == "Qdrant Cloud hybrid":
        return HybridSearchIndex(chunks, semantic_backend="Qdrant Cloud", qdrant_cloud_config=qdrant_cloud_config)
    if backend == "Semantic" or backend == "SentenceTransformer":
        try:
            return SentenceTransformerSearchIndex(chunks)
        except Exception:
            return TfidfSearchIndex(chunks)
    if backend == "Hybrid":
        return HybridSearchIndex(chunks)
    if backend == "Parent-child" and allow_parent_child:
        return ParentChildSearchIndex(chunks, base_backend="Hybrid")
    return TfidfSearchIndex(chunks)


def _merge_score(
    fused: dict[str, tuple[DocumentChunk, float]],
    result: SearchResult,
    weight: float,
) -> None:
    key = result.chunk.chunk_id
    _chunk, current = fused.get(key, (result.chunk, 0.0))
    fused[key] = (result.chunk, current + weight * result.score)


def _combine_parent_chunks(parent_id: str, chunks: list[DocumentChunk]) -> DocumentChunk:
    ordered = sorted(chunks, key=lambda chunk: chunk.chunk_id)
    first = ordered[0]
    text = "\n\n".join(f"[{chunk.chunk_id}] {chunk.text}" for chunk in ordered)
    return DocumentChunk(
        chunk_id=parent_id,
        text=text,
        source_name=first.source_name,
        modality=first.modality,
        timestamp=first.timestamp,
        page_or_frame=first.page_or_frame,
        metadata={
            **first.metadata,
            "parent_id": parent_id,
            "child_count": len(ordered),
            "child_ids": [chunk.chunk_id for chunk in ordered],
            "retrieval_expansion": "parent_child",
        },
    )


def _payload_for_chunk(chunk: DocumentChunk) -> dict:
    return {
        "chunk_id": chunk.chunk_id,
        "parent_id": chunk.metadata.get("parent_id"),
        "source_name": chunk.source_name,
        "modality": chunk.modality,
        "timestamp": chunk.timestamp,
        "page_or_frame": chunk.page_or_frame,
        "text": chunk.text,
        "metadata": {
            key: value
            for key, value in chunk.metadata.items()
            if key != "visual_vector"
        },
    }


def _chunk_from_payload(payload: dict) -> DocumentChunk:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    return DocumentChunk(
        chunk_id=str(payload.get("chunk_id") or ""),
        text=str(payload.get("text") or ""),
        source_name=str(payload.get("source_name") or "qdrant_cloud"),
        modality=str(payload.get("modality") or "text"),
        timestamp=payload.get("timestamp"),
        page_or_frame=payload.get("page_or_frame"),
        metadata=metadata,
    )


def _qdrant_search(client, collection_name: str, query_vector: list[float], limit: int):
    if hasattr(client, "search"):
        return client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit,
        )
    response = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
        with_vectors=False,
    )
    return getattr(response, "points", response)


def _batched(items: list, batch_size: int):
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def _point_id(chunk: DocumentChunk) -> str:
    return qdrant_point_id(chunk)


def qdrant_point_id(chunk: DocumentChunk) -> str:
    basis = "|".join([chunk.source_name, chunk.chunk_id, chunk.text[:200]])
    return str(uuid.uuid5(uuid.NAMESPACE_URL, basis))


def load_chunks_from_qdrant_cloud(
    url: str,
    api_key: str,
    collection_name: str,
    *,
    batch_size: int = 256,
) -> list[DocumentChunk]:
    from qdrant_client import QdrantClient

    if not url.strip() or not api_key.strip():
        raise ValueError("Qdrant Cloud URL and API key are required.")
    if not collection_name.strip():
        raise ValueError("Qdrant Cloud collection name is required.")

    client = QdrantClient(url=url.strip(), api_key=api_key.strip(), timeout=60)
    chunks: list[DocumentChunk] = []
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=collection_name.strip(),
            limit=batch_size,
            with_payload=True,
            with_vectors=False,
            offset=offset,
        )
        for point in points:
            payload = point.payload or {}
            chunk = _chunk_from_payload(payload)
            if chunk.text.strip():
                chunks.append(chunk)
        if offset is None:
            break
    return chunks
