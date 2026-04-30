from __future__ import annotations

from dataclasses import dataclass

from .chunking import DocumentChunk


@dataclass(frozen=True)
class PackedChunk:
    chunk: DocumentChunk
    score: float
    packed_text: str
    estimated_tokens: int


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def pack_context(
    chunks_with_scores: list[tuple[DocumentChunk, float]],
    token_budget: int,
    reserved_tokens: int = 300,
    dedupe_visual_groups: bool = False,
    compact_visual_chunks: bool = False,
) -> list[PackedChunk]:
    available = max(0, token_budget - reserved_tokens)
    packed: list[PackedChunk] = []
    used = 0
    seen_visual_groups: set[str] = set()
    for chunk, score in chunks_with_scores:
        if dedupe_visual_groups:
            group_key = _visual_group_key(chunk)
            if group_key is not None:
                if group_key in seen_visual_groups:
                    continue
                seen_visual_groups.add(group_key)
        header = (
            f"[{chunk.chunk_id}] source={chunk.source_name} "
            f"modality={chunk.modality} "
            f"timestamp={chunk.timestamp or 'None'} "
            f"frame={chunk.page_or_frame or 'None'}\n"
        )
        remaining = available - used - estimate_tokens(header)
        if remaining <= 0:
            break
        body_text = _compact_visual_text(chunk.text) if compact_visual_chunks else chunk.text
        text = _trim_to_token_budget(body_text, remaining)
        token_count = estimate_tokens(header + text)
        if token_count <= 0:
            continue
        packed.append(PackedChunk(chunk=chunk, score=score, packed_text=text, estimated_tokens=token_count))
        used += token_count
    return packed


def packed_context_block(packed_chunks: list[PackedChunk]) -> str:
    blocks = []
    for packed in packed_chunks:
        chunk = packed.chunk
        blocks.append(
            f"[{chunk.chunk_id}] source={chunk.source_name} "
            f"modality={chunk.modality} "
            f"timestamp={chunk.timestamp or 'None'} "
            f"frame={chunk.page_or_frame or 'None'}\n"
            f"{packed.packed_text}"
        )
    return "\n\n".join(blocks)


def _trim_to_token_budget(text: str, token_budget: int) -> str:
    if estimate_tokens(text) <= token_budget:
        return text
    character_budget = max(0, token_budget * 4)
    return text[:character_budget].rstrip() + "\n[trimmed to fit context budget]"


def _visual_group_key(chunk: DocumentChunk) -> str | None:
    if chunk.modality not in {"image", "visual_page", "pdf", "slide", "video"}:
        return None
    page_id = chunk.metadata.get("page_id")
    if page_id:
        return f"page:{page_id}"
    eval_doc_id = chunk.metadata.get("eval_doc_id")
    if eval_doc_id:
        return f"eval:{eval_doc_id}"
    page_num = chunk.metadata.get("page_num") or chunk.page_or_frame or ""
    return f"source:{chunk.source_name}|page:{page_num}"


def _compact_visual_text(text: str) -> str:
    lines = [line.strip() for line in (text or "").splitlines() if line.strip()]
    if not lines:
        return text
    prioritized: list[str] = []
    fallback: list[str] = []
    for line in lines:
        lowered = line.lower()
        if lowered.startswith("ocr text:"):
            prioritized.append(line)
        elif lowered.startswith("manual visual description:"):
            prioritized.append(line)
        elif lowered.startswith("vlm segment description:"):
            fallback.append(line)
        elif "encoder" in lowered or "visual" in lowered or "color" in lowered or "edge" in lowered:
            continue
        else:
            fallback.append(line)
    chosen = prioritized if prioritized else fallback[:2]
    if not chosen:
        chosen = lines[:2]
    return "\n".join(chosen)
