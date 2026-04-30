from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
from pypdf import PdfReader

from .chunking import DocumentChunk, split_text
from .encoders import VisualFeatureEncoder, text_encoder_metadata
from .ocr import OCR_ENGINE, extract_image_text
from .vlm_openai import describe_image_with_gpt5_nano

TEXT_COLUMNS = ("text", "transcript", "content", "description", "notes")


def read_text_bytes(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("utf-8", errors="replace")


def chunks_from_text_bytes(data: bytes, source_name: str, modality: str = "text") -> list[DocumentChunk]:
    return split_text(
        read_text_bytes(data),
        source_name=source_name,
        modality=modality,
        metadata={
            "converter": "utf8_text_reader",
            "chunker": "overlapping_character_text",
            **text_encoder_metadata(),
        },
    )


def chunks_from_csv_bytes(data: bytes, source_name: str) -> list[DocumentChunk]:
    frame = pd.read_csv(BytesIO(data))
    if frame.empty:
        return []

    lower_to_original = {column.lower(): column for column in frame.columns}
    selected = [lower_to_original[column] for column in TEXT_COLUMNS if column in lower_to_original]
    if not selected:
        selected = [
            column
            for column in frame.columns
            if pd.api.types.is_string_dtype(frame[column]) or frame[column].dtype == object
        ]

    chunks: list[DocumentChunk] = []
    for row_number, row in frame.iterrows():
        text = " ".join(str(row[column]) for column in selected if pd.notna(row[column])).strip()
        if text:
            chunks.extend(
                split_text(
                    text,
                    source_name=source_name,
                    modality="text",
                    metadata={
                        "row": int(row_number) + 1,
                        "converter": "pandas_csv_text_columns",
                        "chunker": "csv_row_text",
                        **text_encoder_metadata(),
                    },
                )
            )
    return chunks


def chunks_from_pdf_bytes(data: bytes, source_name: str) -> list[DocumentChunk]:
    reader = PdfReader(BytesIO(data))
    chunks: list[DocumentChunk] = []
    for page_index, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_chunks = split_text(
            text,
            source_name=source_name,
            modality="pdf",
            metadata={
                "page": page_index,
                "converter": "pypdf_page_text",
                "chunker": "pdf_page_overlapping_text",
                **text_encoder_metadata(),
            },
        )
        for chunk in page_chunks:
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_name=chunk.source_name,
                    modality=chunk.modality,
                    page_or_frame=f"page {page_index}",
                    metadata=chunk.metadata,
                )
            )
    return chunks


def chunk_from_image_bytes(
    data: bytes,
    source_name: str,
    description: str = "",
    openai_api_key: str | None = None,
) -> DocumentChunk:
    encoding = VisualFeatureEncoder().encode_bytes(data)
    image = None
    ocr_text = ""
    ocr_status = ""
    vlm_text = ""
    vlm_status = "OpenAI VLM skipped: image could not be loaded."
    try:
        from PIL import Image

        image = Image.open(BytesIO(data)).convert("RGB")
        ocr_text, ocr_status = extract_image_text(image)
        vlm_text, vlm_status = describe_image_with_gpt5_nano(
            image,
            openai_api_key,
            context=f"source={source_name}; OCR text={ocr_text}",
        )
    except Exception as exc:
        ocr_status = f"OCR failed before extraction: {exc}"
    text = _visual_chunk_text(description, encoding.summary, ocr_text=ocr_text, vlm_text=vlm_text)
    return DocumentChunk(
        chunk_id="",
        text=text,
        source_name=source_name,
        modality="image",
        metadata={
            "converter": "pil_image_loader",
            "chunker": "single_image_asset",
            "encoder": encoding.encoder_name,
            "embedding_space": "visual_stats",
            "visual_vector": encoding.vector,
            "manual_description": description.strip(),
            "ocr_engine": OCR_ENGINE,
            "ocr_status": ocr_status,
            "vlm_engine": "openai.gpt-5-nano.image_input",
            "vlm_status": vlm_status,
        },
    )


def chunk_from_video_frame(
    source_name: str,
    frame_image,
    timestamp: str,
    frame_number: str,
    description: str = "",
    openai_api_key: str | None = None,
) -> DocumentChunk:
    encoding = VisualFeatureEncoder().encode_image(frame_image)
    ocr_text, ocr_status = extract_image_text(frame_image)
    vlm_text, vlm_status = describe_image_with_gpt5_nano(
        frame_image,
        openai_api_key,
        context=f"source={source_name}; segment={timestamp}; frame={frame_number}; OCR text={ocr_text}",
    )
    text = _visual_chunk_text(description, encoding.summary, ocr_text=ocr_text, vlm_text=vlm_text)
    return DocumentChunk(
        chunk_id="",
        text=text,
        source_name=source_name,
        modality="video",
        timestamp=timestamp,
        page_or_frame=frame_number,
        metadata={
            "converter": "opencv_frame_sampler",
            "chunker": "single_video_frame",
            "encoder": encoding.encoder_name,
            "embedding_space": "visual_stats",
            "visual_vector": encoding.vector,
            "manual_description": description.strip(),
            "ocr_engine": OCR_ENGINE,
            "ocr_status": ocr_status,
            "vlm_engine": "openai.gpt-5-nano.image_input",
            "vlm_status": vlm_status,
        },
    )


def chunks_from_descriptions(
    descriptions: list[dict[str, str]],
    source_name: str,
    modality: str,
) -> list[DocumentChunk]:
    chunks = []
    for item in descriptions:
        description = (item.get("description") or "").strip()
        if not description:
            continue
        chunks.append(
            DocumentChunk(
                chunk_id="",
                text=description,
                source_name=item.get("source_name") or source_name,
                modality=item.get("modality") or modality,
                timestamp=item.get("timestamp") or None,
                page_or_frame=item.get("page_or_frame") or None,
                metadata={
                    "converter": item.get("converter") or "prewritten_description",
                    "chunker": item.get("chunker") or "single_description_record",
                    **text_encoder_metadata(),
                    **{key: value for key, value in item.items() if key not in {"description"}},
                },
            )
        )
    return chunks


def _visual_chunk_text(
    description: str,
    encoder_summary: str,
    ocr_text: str = "",
    vlm_text: str = "",
) -> str:
    parts = []
    description = description.strip()
    if description:
        parts.append(f"Manual visual description: {description}")
    if ocr_text.strip():
        parts.append(f"OCR text: {ocr_text.strip()}")
    if vlm_text.strip():
        parts.append(f"VLM segment description: {vlm_text.strip()}")
    if not parts:
        parts.append("No manual visual description, OCR text, or VLM description was available.")
    parts.append(encoder_summary)
    return "\n".join(parts)


def load_sample_chunks(sample_dir: Path) -> list[DocumentChunk]:
    chunks: list[DocumentChunk] = []
    chunks.extend(
        split_text(
            (sample_dir / "meeting_transcript.txt").read_text(encoding="utf-8"),
            source_name="meeting_transcript.txt",
            modality="audio_transcript",
            metadata={
                "converter": "sample_audio_transcript_text",
                "chunker": "overlapping_character_text",
                **text_encoder_metadata(),
            },
        )
    )
    chunks.extend(
        split_text(
            (sample_dir / "slide_notes.txt").read_text(encoding="utf-8"),
            source_name="slide_notes.txt",
            modality="slide",
            metadata={
                "converter": "sample_slide_notes_text",
                "chunker": "overlapping_character_text",
                **text_encoder_metadata(),
            },
        )
    )
    descriptions = pd.read_csv(sample_dir / "image_descriptions.csv").to_dict("records")
    chunks.extend(chunks_from_descriptions(descriptions, "image_descriptions.csv", "image"))
    return chunks
