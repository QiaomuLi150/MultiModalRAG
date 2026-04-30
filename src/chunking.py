from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from typing import Iterable


@dataclass(frozen=True)
class DocumentChunk:
    chunk_id: str
    text: str
    source_name: str
    modality: str
    timestamp: str | None = None
    page_or_frame: str | None = None
    metadata: dict = field(default_factory=dict)


def split_text(
    text: str,
    source_name: str,
    modality: str = "text",
    chunk_size: int = 900,
    overlap: int = 120,
    metadata: dict | None = None,
) -> list[DocumentChunk]:
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return []
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks: list[DocumentChunk] = []
    start = 0
    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunks.append(
            DocumentChunk(
                chunk_id="",
                text=cleaned[start:end],
                source_name=source_name,
                modality=modality,
                metadata=dict(metadata or {}),
            )
        )
        if end == len(cleaned):
            break
        start = max(0, end - overlap)
    return chunks


def assign_chunk_ids(chunks: Iterable[DocumentChunk]) -> list[DocumentChunk]:
    return [
        replace(chunk, chunk_id=f"chunk_{index:03d}")
        for index, chunk in enumerate(chunks, start=1)
    ]


def parent_id_for_chunk(chunk: DocumentChunk) -> str:
    explicit = chunk.metadata.get("parent_id")
    if explicit:
        return str(explicit)
    basis = "|".join(
        [
            chunk.source_name,
            chunk.modality,
            chunk.timestamp or "",
            chunk.page_or_frame or "",
            str(chunk.metadata.get("page", "")),
            str(chunk.metadata.get("row", "")),
        ]
    )
    digest = hashlib.sha1(basis.encode("utf-8")).hexdigest()[:10]
    return f"parent_{digest}"


def add_parent_metadata(chunks: Iterable[DocumentChunk]) -> list[DocumentChunk]:
    enriched = []
    for chunk in chunks:
        metadata = dict(chunk.metadata)
        metadata.setdefault("parent_id", parent_id_for_chunk(chunk))
        metadata.setdefault("child_id", chunk.chunk_id)
        enriched.append(replace(chunk, metadata=metadata))
    return enriched


def text_preview(text: str, limit: int = 180) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."
