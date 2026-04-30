from __future__ import annotations

from dataclasses import dataclass

from .retrieval import SearchResult


@dataclass(frozen=True)
class RerankerSpec:
    key: str
    label: str
    model_name: str | None
    note: str


RERANKER_SPECS = (
    RerankerSpec("off", "Off", None, "Fastest. No reranking is applied."),
    RerankerSpec(
        "minilm_l2",
        "MiniLM-L2",
        "cross-encoder/ms-marco-MiniLM-L2-v2",
        "Very fast. Smallest quality boost.",
    ),
    RerankerSpec(
        "minilm_l4",
        "MiniLM-L4",
        "cross-encoder/ms-marco-MiniLM-L4-v2",
        "Balanced speed and ranking quality. Recommended.",
    ),
    RerankerSpec(
        "minilm_l6",
        "MiniLM-L6",
        "cross-encoder/ms-marco-MiniLM-L6-v2",
        "Strongest lightweight reranker. Higher latency.",
    ),
)


def reranker_choices() -> list[str]:
    return [spec.label for spec in RERANKER_SPECS]


def reranker_by_label(label: str) -> RerankerSpec:
    for spec in RERANKER_SPECS:
        if spec.label == label:
            return spec
    raise KeyError(f"Unknown reranker: {label}")


class CrossEncoderReranker:
    def __init__(self, model_name: str):
        from sentence_transformers.cross_encoder import CrossEncoder

        self.model_name = model_name
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, results: list[SearchResult], top_n: int) -> list[SearchResult]:
        if not results:
            return []
        limited = results[: max(1, top_n)]
        pairs = [(query, result.chunk.text) for result in limited]
        scores = self.model.predict(pairs)
        reranked = [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in zip(limited, scores)
        ]
        reranked = sorted(reranked, key=lambda item: item.score, reverse=True)
        if len(results) > len(limited):
            reranked.extend(results[len(limited) :])
        return reranked

