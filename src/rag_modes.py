from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .chunking import DocumentChunk
from .rerankers import CrossEncoderReranker, RerankerSpec
from .retrieval import SearchResult, build_search_index
from .schemas import PageRecord
from .visual_retrieval import build_visual_search_index


VISUAL_MODALITIES = {"pdf", "slide", "image", "video"}


@dataclass(frozen=True)
class RagModeSpec:
    family: str
    label: str
    description: str
    text_backend: str
    use_step_back: bool = False
    use_page_assets: bool = False
    use_visual_backend: bool = False
    use_text_hybrid: bool = False
    use_rerank: bool = False


MODE_SPECS = (
    RagModeSpec("Text", "Text TF-IDF", "Lexical retrieval over chunk text.", "TF-IDF"),
    RagModeSpec("Text", "Text Semantic", "Dense text retrieval over chunk text.", "Semantic"),
    RagModeSpec("Text", "Text Hybrid", "Dense plus TF-IDF retrieval over chunk text.", "Hybrid"),
    RagModeSpec(
        "Text",
        "Text Hybrid + Step-back",
        "Hybrid chunk retrieval with query expansion.",
        "Hybrid",
        use_step_back=True,
    ),
    RagModeSpec(
        "Text",
        "Parent-child",
        "Retrieve child chunks and expand to parent sections/pages.",
        "Parent-child",
    ),
    RagModeSpec(
        "Visual",
        "Visual Page",
        "Page-oriented retrieval over rendered page images with multivector visual search.",
        "Hybrid",
        use_page_assets=True,
        use_visual_backend=True,
    ),
    RagModeSpec(
        "Visual",
        "Visual Page + OCR",
        "Multivector visual retrieval over rendered page images, with OCR attached for evidence.",
        "Hybrid",
        use_page_assets=True,
        use_visual_backend=True,
    ),
    RagModeSpec(
        "Hybrid",
        "Visual Page + Text Hybrid",
        "Fuse multivector visual page retrieval with chunk-level text retrieval.",
        "Hybrid",
        use_page_assets=True,
        use_visual_backend=True,
        use_text_hybrid=True,
    ),
    RagModeSpec(
        "Hybrid",
        "Visual Page + Text Hybrid + Rerank",
        "Fuse multivector visual page retrieval with chunk text, then rerank by query overlap.",
        "Hybrid",
        use_page_assets=True,
        use_visual_backend=True,
        use_text_hybrid=True,
        use_rerank=True,
    ),
)


def families() -> list[str]:
    return sorted({mode.family for mode in MODE_SPECS}, key=lambda item: ("Text", "Visual", "Hybrid").index(item))


def modes_for_family(family: str) -> list[RagModeSpec]:
    return [mode for mode in MODE_SPECS if mode.family == family]


def mode_by_label(label: str) -> RagModeSpec:
    for mode in MODE_SPECS:
        if mode.label == label:
            return mode
    raise KeyError(f"Unknown RAG mode: {label}")


class ModeSearchIndex:
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        raise NotImplementedError

    def status(self) -> dict:
        return {"encoder_name": getattr(self, "encoder_name", "")}


class SingleIndexAdapter(ModeSearchIndex):
    def __init__(self, index, reranker: CrossEncoderReranker | None = None, rerank_top_n: int = 8):
        self.index = index
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.encoder_name = getattr(index, "encoder_name", "single_index")

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        results = self.index.search(query, top_k=max(top_k, self.rerank_top_n if self.reranker else top_k))
        if self.reranker is not None:
            results = self.reranker.rerank(query, results, top_n=self.rerank_top_n)
        return results[:top_k]

    def status(self) -> dict:
        if hasattr(self.index, "status"):
            return self.index.status()
        return {"point_count": 0, "encoder_name": self.encoder_name}


class HybridModeSearchIndex(ModeSearchIndex):
    def __init__(
        self,
        primary,
        secondary,
        use_rerank: bool = False,
        reranker: CrossEncoderReranker | None = None,
        rerank_top_n: int = 8,
    ):
        self.primary = primary
        self.secondary = secondary
        self.use_rerank = use_rerank
        self.reranker = reranker
        self.rerank_top_n = rerank_top_n
        self.encoder_name = (
            f"mode_hybrid({getattr(primary, 'encoder_name', 'primary')}+"
            f"{getattr(secondary, 'encoder_name', 'secondary')})"
        )

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        primary_results = self.primary.search(query, top_k=max(top_k * 3, top_k))
        secondary_results = self.secondary.search(query, top_k=max(top_k * 3, top_k))
        merged = reciprocal_rank_fusion([primary_results, secondary_results], top_k=max(top_k * 3, top_k))
        if self.reranker is not None:
            merged = self.reranker.rerank(query, merged, top_n=self.rerank_top_n)
        elif self.use_rerank:
            merged = rerank_by_query_overlap(query, merged)
        return merged[:top_k]

    def status(self) -> dict:
        return {
            "encoder_name": self.encoder_name,
            "visual_or_primary": _status_from_index(self.primary),
            "text_or_secondary": _status_from_index(self.secondary),
            "point_count": (
                _status_from_index(self.primary).get("point_count", 0)
                + _status_from_index(self.secondary).get("point_count", 0)
            ),
        }


def build_mode_index(
    chunks: list[DocumentChunk],
    mode: RagModeSpec,
    search_backend: str,
    qdrant_cloud_config: dict | None = None,
    page_records: list[PageRecord] | None = None,
    visual_model_name: str = "vidore/colqwen2-v1.0-hf",
    visual_local_files_only: bool = True,
    visual_embedding_batch_size: int = 1,
    light_compression_mode: str = "none",
    light_target_tokens: int | None = None,
    use_muvera_proxy: bool = False,
    muvera_candidate_count: int = 12,
    reranker_spec: RerankerSpec | None = None,
    rerank_top_n: int = 8,
):
    reranker = _build_reranker(reranker_spec)
    text_index = build_search_index(chunks, search_backend, qdrant_cloud_config=qdrant_cloud_config)
    if not mode.use_page_assets and not mode.use_text_hybrid and not mode.use_visual_backend:
        return SingleIndexAdapter(text_index, reranker=reranker, rerank_top_n=rerank_top_n)

    page_index = None
    if mode.use_visual_backend and (page_records or qdrant_cloud_config is not None):
        page_index = build_visual_search_index(
            page_records or [],
            model_name=visual_model_name,
            local_files_only=visual_local_files_only,
            embedding_batch_size=visual_embedding_batch_size,
            light_compression_mode=light_compression_mode,
            light_target_tokens=light_target_tokens,
            use_muvera_proxy=use_muvera_proxy,
            muvera_candidate_count=muvera_candidate_count,
            qdrant_cloud_config=qdrant_cloud_config,
        )
    if page_index is None:
        page_chunks = build_page_asset_chunks(chunks)
        page_backend = _page_backend(search_backend)
        page_index = build_search_index(page_chunks, page_backend, qdrant_cloud_config=qdrant_cloud_config)
    if mode.use_text_hybrid:
        return HybridModeSearchIndex(
            page_index,
            text_index,
            use_rerank=mode.use_rerank,
            reranker=reranker,
            rerank_top_n=rerank_top_n,
        )
    return SingleIndexAdapter(page_index, reranker=reranker, rerank_top_n=rerank_top_n)


def build_page_asset_chunks(chunks: Iterable[DocumentChunk]) -> list[DocumentChunk]:
    grouped: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        if chunk.modality not in VISUAL_MODALITIES:
            continue
        parent_id = str(chunk.metadata.get("parent_id") or chunk.chunk_id)
        grouped.setdefault(parent_id, []).append(chunk)

    page_chunks: list[DocumentChunk] = []
    for parent_id, group in grouped.items():
        ordered = sorted(group, key=lambda item: item.chunk_id)
        first = ordered[0]
        combined_text = "\n\n".join(item.text for item in ordered if item.text.strip())
        if not combined_text.strip():
            continue
        metadata = dict(first.metadata)
        metadata.update(
            {
                "parent_id": parent_id,
                "child_count": len(ordered),
                "child_ids": [item.chunk_id for item in ordered],
                "retrieval_granularity": "page_asset",
            }
        )
        page_chunks.append(
            DocumentChunk(
                chunk_id=parent_id,
                text=combined_text,
                source_name=first.source_name,
                modality=first.modality,
                timestamp=first.timestamp,
                page_or_frame=first.page_or_frame,
                metadata=metadata,
            )
        )
    return page_chunks


def reciprocal_rank_fusion(result_lists: list[list[SearchResult]], top_k: int, k: int = 60) -> list[SearchResult]:
    fused: dict[str, tuple[DocumentChunk, float]] = {}
    for results in result_lists:
        for rank, result in enumerate(results, start=1):
            weight = 1.0 / (k + rank)
            current_chunk, current_score = fused.get(result.chunk.chunk_id, (result.chunk, 0.0))
            fused[result.chunk.chunk_id] = (current_chunk, current_score + weight)
    ranked = sorted(fused.values(), key=lambda item: item[1], reverse=True)
    return [SearchResult(chunk=chunk, score=score) for chunk, score in ranked[:top_k]]


def rerank_by_query_overlap(query: str, results: list[SearchResult]) -> list[SearchResult]:
    query_terms = {term for term in query.lower().split() if term}
    if not query_terms:
        return results

    reranked: list[SearchResult] = []
    for result in results:
        text_terms = set(result.chunk.text.lower().split())
        overlap = len(query_terms & text_terms)
        reranked.append(SearchResult(chunk=result.chunk, score=result.score + overlap * 0.05))
    return sorted(reranked, key=lambda item: item.score, reverse=True)


def _page_backend(search_backend: str) -> str:
    if search_backend == "Parent-child":
        return "Hybrid"
    return search_backend


def _status_from_index(index) -> dict:
    if hasattr(index, "status"):
        try:
            return index.status()
        except Exception as exc:
            return {"status_error": str(exc), "encoder_name": getattr(index, "encoder_name", "")}
    return {"encoder_name": getattr(index, "encoder_name", "")}


def _build_reranker(reranker_spec: RerankerSpec | None) -> CrossEncoderReranker | None:
    if reranker_spec is None or not reranker_spec.model_name:
        return None
    return CrossEncoderReranker(reranker_spec.model_name)
