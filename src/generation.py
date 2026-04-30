from __future__ import annotations

import base64
from io import BytesIO
import os
from typing import Any

from PIL import Image

from .chunking import DocumentChunk
from .context import pack_context, packed_context_block

MODEL_NAME = "gpt-5-nano"

SYSTEM_PROMPT = """You are a careful research assistant.
Answer only using the provided context chunks.
If the context is insufficient, say what is missing.
Cite sources using chunk IDs like [chunk_003].
For short extractive questions, return the shortest exact answer span supported by the context before the citation.
Do not invent facts."""

STEPBACK_SYSTEM_PROMPT = """You rewrite user questions for retrieval.
Return one broader, more general search question that could retrieve useful background context.
Do not answer the question.
Return only the rewritten question."""

MULTIMODAL_SYSTEM_PROMPT = """You are a careful multimodal research assistant.
Use the provided page images and supporting text evidence to answer the question.
For short extractive questions, return the shortest exact answer span supported by the evidence.
If the evidence is insufficient or unreadable, say so directly.
Cite the supporting page or chunk IDs like [doc_page_001] or [chunk_003].
Do not invent facts."""


def get_api_key(secrets: Any | None = None, user_api_key: str | None = None) -> str | None:
    if user_api_key and user_api_key.strip():
        return user_api_key.strip()
    if secrets is not None:
        try:
            key = secrets.get("OPENAI_API_KEY")
            if key:
                return str(key)
        except Exception:
            pass
    return os.environ.get("OPENAI_API_KEY")


def build_user_prompt(
    question: str,
    chunks_with_scores: list[tuple[DocumentChunk, float]],
    token_budget: int = 3000,
    supplemental_context: str | None = None,
) -> str:
    use_visual_refinement = any(
        chunk.modality in {"image", "visual_page", "pdf", "slide", "video"}
        for chunk, _score in chunks_with_scores
    )
    packed_chunks = pack_context(
        chunks_with_scores,
        token_budget=token_budget,
        dedupe_visual_groups=use_visual_refinement,
        compact_visual_chunks=use_visual_refinement,
    )
    graph_section = ""
    if supplemental_context and supplemental_context.strip():
        graph_section = f"\n\nGraph context:\n{supplemental_context.strip()}"
    return f"""Question:
{question}

Context chunks:
{packed_context_block(packed_chunks)}
{graph_section}

Answer with citations:"""


def generate_answer(
    question: str,
    chunks_with_scores: list[tuple[DocumentChunk, float]],
    api_key: str,
    token_budget: int = 3000,
    supplemental_context: str | None = None,
) -> str:
    from openai import OpenAI

    visual_chunks = _unique_visual_chunks(chunks_with_scores)
    if visual_chunks:
        return generate_multimodal_answer(
            question,
            chunks_with_scores,
            visual_chunks,
            api_key,
            token_budget=token_budget,
            supplemental_context=supplemental_context,
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": build_user_prompt(
                    question,
                    chunks_with_scores,
                    token_budget,
                    supplemental_context=supplemental_context,
                ),
            },
        ],
    )
    return response.output_text


def generate_multimodal_answer(
    question: str,
    chunks_with_scores: list[tuple[DocumentChunk, float]],
    visual_chunks: list[DocumentChunk],
    api_key: str,
    token_budget: int = 3000,
    supplemental_context: str | None = None,
) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    content: list[dict[str, str]] = [
        {
            "type": "input_text",
            "text": build_multimodal_user_prompt(
                question,
                chunks_with_scores,
                visual_chunks,
                token_budget=token_budget,
                supplemental_context=supplemental_context,
            ),
        }
    ]
    for chunk in visual_chunks:
        content.append(
            {
                "type": "input_text",
                "text": (
                    f"Image evidence [{chunk.metadata.get('page_id') or chunk.chunk_id}] "
                    f"source={chunk.source_name} page={chunk.metadata.get('page_num') or chunk.page_or_frame or 'unknown'}"
                ),
            }
        )
        content.append(
            {
                "type": "input_image",
                "image_url": _image_path_data_url(str(chunk.metadata.get("image_path") or "")),
            }
        )
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": MULTIMODAL_SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
    )
    return response.output_text


def generate_stepback_question(question: str, api_key: str) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=MODEL_NAME,
        input=[
            {"role": "system", "content": STEPBACK_SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    return response.output_text.strip()


def build_multimodal_user_prompt(
    question: str,
    chunks_with_scores: list[tuple[DocumentChunk, float]],
    visual_chunks: list[DocumentChunk],
    token_budget: int,
    supplemental_context: str | None = None,
) -> str:
    visual_ids = ", ".join(
        str(chunk.metadata.get("page_id") or chunk.chunk_id)
        for chunk in visual_chunks
    )
    non_visual = [(chunk, score) for chunk, score in chunks_with_scores if chunk not in visual_chunks]
    packed_text = pack_context(non_visual, token_budget=max(800, token_budget // 2))
    visual_summaries = []
    for chunk in visual_chunks:
        ocr_text = str(chunk.metadata.get("ocr_text") or "").strip()
        title_hint = str(chunk.metadata.get("title_hint") or "").strip()
        page_id = str(chunk.metadata.get("page_id") or chunk.chunk_id)
        page_num = str(chunk.metadata.get("page_num") or chunk.page_or_frame or "")
        lines = [f"[{page_id}] source={chunk.source_name} page={page_num or 'unknown'}"]
        if title_hint:
            lines.append(f"title_hint={title_hint}")
        if ocr_text:
            lines.append(f"ocr_snippet={_trim_for_prompt(ocr_text, 500)}")
        visual_summaries.append("\n".join(lines))
    graph_section = ""
    if supplemental_context and supplemental_context.strip():
        graph_section = f"\n\nSupplemental context:\n{supplemental_context.strip()}"
    text_section = packed_context_block(packed_text) if packed_text else "No additional text chunks."
    return (
        f"Question:\n{question}\n\n"
        f"Primary visual evidence IDs:\n{visual_ids}\n\n"
        f"Visual evidence summaries:\n{'\n\n'.join(visual_summaries)}\n\n"
        f"Additional text evidence:\n{text_section}"
        f"{graph_section}\n\n"
        "Answer with a short grounded response and citations."
    )


def _unique_visual_chunks(chunks_with_scores: list[tuple[DocumentChunk, float]], limit: int = 3) -> list[DocumentChunk]:
    selected: list[DocumentChunk] = []
    seen: set[str] = set()
    for chunk, _score in chunks_with_scores:
        image_path = str(chunk.metadata.get("image_path") or "")
        if chunk.modality not in {"image", "visual_page", "pdf", "slide", "video"}:
            continue
        if not image_path or not os.path.exists(image_path):
            continue
        key = str(chunk.metadata.get("page_id") or chunk.chunk_id or image_path)
        if key in seen:
            continue
        seen.add(key)
        selected.append(chunk)
        if len(selected) >= limit:
            break
    return selected


def _image_path_data_url(image_path: str) -> str:
    with Image.open(image_path) as image:
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _trim_for_prompt(text: str, max_chars: int) -> str:
    text = " ".join((text or "").split())
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."
