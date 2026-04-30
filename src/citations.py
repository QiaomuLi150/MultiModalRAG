from __future__ import annotations

from .chunking import DocumentChunk, text_preview
from .context import PackedChunk


def citation_label(chunk: DocumentChunk) -> str:
    return f"[{chunk.chunk_id}]"


def format_evidence(chunk: DocumentChunk, score: float | None = None) -> str:
    parts = [
        citation_label(chunk),
        f"source={chunk.source_name}",
        f"modality={chunk.modality}",
    ]
    if chunk.timestamp:
        parts.append(f"timestamp={chunk.timestamp}")
    if chunk.page_or_frame:
        parts.append(f"frame={chunk.page_or_frame}")
    if score is not None:
        parts.append(f"score={score:.3f}")
    return " ".join(parts) + "\n" + text_preview(chunk.text, 500)


def context_block(chunks_with_scores: list[tuple[DocumentChunk, float]]) -> str:
    blocks = []
    for chunk, _score in chunks_with_scores:
        timestamp = chunk.timestamp or "None"
        frame = chunk.page_or_frame or "None"
        blocks.append(
            f"[{chunk.chunk_id}] source={chunk.source_name} "
            f"modality={chunk.modality} timestamp={timestamp} frame={frame}\n"
            f"{chunk.text}"
        )
    return "\n\n".join(blocks)


def context_block_from_packed(packed_chunks: list[PackedChunk]) -> str:
    blocks = []
    for packed in packed_chunks:
        chunk = packed.chunk
        timestamp = chunk.timestamp or "None"
        frame = chunk.page_or_frame or "None"
        blocks.append(
            f"[{chunk.chunk_id}] source={chunk.source_name} "
            f"modality={chunk.modality} timestamp={timestamp} frame={frame}\n"
            f"{packed.packed_text}"
        )
    return "\n\n".join(blocks)
