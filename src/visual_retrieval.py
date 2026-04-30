from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np

from .chunking import DocumentChunk
from .embeddings.image_retriever import ColQwen2ImageRetriever, VisualRetrieverUnavailable
from .embeddings.light_multivector import compress_multivector
from .embeddings.muvera_adapter import build_proxy_vector, exact_maxsim_score
from .retrieval import SearchResult
from .schemas import PageRecord
from .storage.qdrant_store import (
    QdrantSingleVectorConfig,
    QdrantSingleVectorStore,
    QdrantVisualConfig,
    QdrantVisualStore,
)


@dataclass(frozen=True)
class VisualRetrievalConfig:
    model_name: str = "vidore/colqwen2-v1.0-hf"
    local_files_only: bool = True
    device: str | None = None
    embedding_batch_size: int = 1
    light_compression_mode: str = "none"
    light_target_tokens: int | None = None
    use_muvera_proxy: bool = False
    muvera_candidate_count: int = 12
    qdrant_collection_name: str = "visual_pages_multivector"
    qdrant_proxy_collection_name: str = "visual_pages_muvera_proxy"
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_local_path: str | None = None


class VisualPageSearchIndex:
    def __init__(self, page_records: list[PageRecord], config: VisualRetrievalConfig):
        self.page_records = page_records
        self.config = config
        self.encoder_name = f"visual_multivector({config.model_name})"
        self.retriever = ColQwen2ImageRetriever(
            model_name=config.model_name,
            device=config.device,
            local_files_only=config.local_files_only,
        )
        self.store = QdrantVisualStore(
            QdrantVisualConfig(
                collection_name=config.qdrant_collection_name,
                url=config.qdrant_url,
                api_key=config.qdrant_api_key,
                local_path=config.qdrant_local_path,
            )
        )
        self.proxy_store = None
        if config.use_muvera_proxy:
            self.proxy_store = QdrantSingleVectorStore(
                QdrantSingleVectorConfig(
                    collection_name=config.qdrant_proxy_collection_name,
                    url=config.qdrant_url,
                    api_key=config.qdrant_api_key,
                    local_path=config.qdrant_local_path,
                )
            )
        self.page_embeddings: dict[str, np.ndarray] = {}
        if page_records:
            self._index_page_records(page_records)

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        if not query.strip():
            return []
        query_vector = self.retriever.encode_query(query)
        hits = self._search_hits(query_vector, top_k=top_k)
        return [
            SearchResult(chunk=_page_payload_to_chunk(payload), score=float(score))
            for payload, score in hits
        ]

    def status(self) -> dict:
        return {
            **self.store.status(),
            "encoder_name": self.encoder_name,
            "visual_model_name": self.config.model_name,
            "local_files_only": self.config.local_files_only,
            "embedding_batch_size": self.config.embedding_batch_size,
            "light_compression_mode": self.config.light_compression_mode,
            "light_target_tokens": self.config.light_target_tokens,
            "use_muvera_proxy": self.config.use_muvera_proxy,
            "muvera_candidate_count": self.config.muvera_candidate_count,
            "proxy_status": self.proxy_store.status() if self.proxy_store is not None else None,
        }

    def _index_page_records(self, page_records: list[PageRecord]) -> None:
        batch_size = max(1, self.config.embedding_batch_size)
        for start in range(0, len(page_records), batch_size):
            batch_records = page_records[start : start + batch_size]
            embeddings = self.retriever.encode_images(
                [page.image_path for page in batch_records],
                batch_size=batch_size,
            )
            embeddings = [
                compress_multivector(
                    embedding,
                    mode=self.config.light_compression_mode,
                    target_tokens=self.config.light_target_tokens,
                )
                for embedding in embeddings
            ]
            self.store.upsert_pages(batch_records, embeddings)
            for page, embedding in zip(batch_records, embeddings):
                self.page_embeddings[page.page_id] = embedding
            if self.proxy_store is not None:
                proxy_vectors = [build_proxy_vector(embedding) for embedding in embeddings]
                self.proxy_store.upsert_pages(batch_records, proxy_vectors)
            self.retriever.clear_device_cache()

    def _search_hits(self, query_vector: np.ndarray, top_k: int):
        if self.proxy_store is None:
            hits = self.store.search(query_vector, top_k=top_k)
            return [(hit.payload or {}, float(hit.score)) for hit in hits]

        candidate_count = max(top_k, self.config.muvera_candidate_count)
        proxy_query = build_proxy_vector(query_vector)
        proxy_hits = self.proxy_store.search(proxy_query, top_k=candidate_count)
        if not self.page_embeddings:
            return [(hit.payload or {}, float(hit.score)) for hit in proxy_hits[:top_k]]

        reranked = []
        for hit in proxy_hits:
            page_id = str(hit.payload.get("page_id") or hit.id)
            page_embedding = self.page_embeddings.get(page_id)
            if page_embedding is None:
                reranked.append((hit.payload or {}, float(hit.score)))
                continue
            score = exact_maxsim_score(query_vector, page_embedding)
            reranked.append((hit.payload or {}, float(score)))
        reranked.sort(key=lambda item: item[1], reverse=True)
        return reranked[:top_k]


def build_visual_search_index(
    page_records: list[PageRecord],
    model_name: str,
    local_files_only: bool,
    embedding_batch_size: int,
    light_compression_mode: str,
    light_target_tokens: int | None,
    use_muvera_proxy: bool,
    muvera_candidate_count: int,
    qdrant_cloud_config: dict | None = None,
) -> VisualPageSearchIndex:
    collection_name = "visual_pages_multivector"
    proxy_collection_name = "visual_pages_muvera_proxy"
    url = None
    api_key = None
    if qdrant_cloud_config:
        base_collection = qdrant_cloud_config.get("collection_name", "multimodal_chunks")
        collection_name = f"{base_collection}_visual_pages_multivector"
        proxy_collection_name = f"{base_collection}_visual_pages_muvera_proxy"
        url = qdrant_cloud_config.get("url")
        api_key = qdrant_cloud_config.get("api_key")
    return VisualPageSearchIndex(
        page_records,
        VisualRetrievalConfig(
            model_name=model_name,
            local_files_only=local_files_only,
            embedding_batch_size=embedding_batch_size,
            light_compression_mode=light_compression_mode,
            light_target_tokens=light_target_tokens,
            use_muvera_proxy=use_muvera_proxy,
            muvera_candidate_count=muvera_candidate_count,
            qdrant_collection_name=collection_name,
            qdrant_proxy_collection_name=proxy_collection_name,
            qdrant_url=url,
            qdrant_api_key=api_key,
        ),
    )


def _page_payload_to_chunk(payload: dict) -> DocumentChunk:
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    ocr_text = str(payload.get("ocr_text") or "").strip()
    title_hint = str(payload.get("title_hint") or "").strip()
    image_path = str(payload.get("image_path") or "")
    source_type = str(payload.get("source_type") or "page")
    parts = [
        f"Page retrieval result: doc_id={payload.get('doc_id', '')} page={payload.get('page_num', '')}.",
    ]
    if title_hint:
        parts.append(f"Title hint: {title_hint}")
    if ocr_text:
        parts.append(f"OCR text: {ocr_text}")
    else:
        parts.append("OCR text was unavailable for this page.")
    if image_path and Path(image_path).exists():
        parts.append(f"Rendered page image: {image_path}")
    metadata = {
        **metadata,
        "doc_id": payload.get("doc_id"),
        "page_id": payload.get("page_id"),
        "page_num": payload.get("page_num"),
        "image_path": image_path,
        "ocr_text": ocr_text,
        "title_hint": title_hint,
        "source_type": source_type,
        "retrieval_granularity": "visual_page",
    }
    return DocumentChunk(
        chunk_id=str(payload.get("page_id") or ""),
        text="\n".join(parts),
        source_name=str(payload.get("source_name") or payload.get("source_path") or "visual_page"),
        modality="visual_page",
        page_or_frame=f"page {payload.get('page_num')}" if payload.get("page_num") else None,
        metadata=metadata,
    )
