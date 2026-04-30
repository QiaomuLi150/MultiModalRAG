from __future__ import annotations

from pathlib import Path
from tempfile import NamedTemporaryFile

from .chunking import DocumentChunk, split_text
from .encoders import text_encoder_metadata

DOCLING_MODES = ("strict", "balanced", "broad")


def chunks_from_pdf_docling_bytes(
    data: bytes,
    source_name: str,
    mode: str = "balanced",
) -> tuple[list[DocumentChunk], str]:
    mode = mode.lower().strip()
    if mode not in DOCLING_MODES:
        raise ValueError(f"Unsupported Docling mode: {mode}")

    try:
        from docling.document_converter import DocumentConverter
        from docling.chunking import HierarchicalChunker
    except Exception as exc:
        return [], f"Docling unavailable: {exc}"

    with NamedTemporaryFile(suffix=Path(source_name).suffix or ".pdf", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        document = DocumentConverter().convert(source=tmp_path).document
        units = [
            chunk.text.strip()
            for chunk in HierarchicalChunker().chunk(dl_doc=document)
            if getattr(chunk, "text", "").strip()
        ]
        sections = _merge_units(units, mode=mode)
        chunks: list[DocumentChunk] = []
        for section_index, section in enumerate(sections, start=1):
            chunks.extend(
                split_text(
                    section,
                    source_name=source_name,
                    modality="pdf",
                    chunk_size=1200,
                    overlap=160,
                    metadata={
                        "section": section_index,
                        "parent_id": f"{source_name}:section:{section_index}",
                        "converter": "docling_hierarchical_pdf",
                        "chunker": f"docling_section_{mode}",
                        **text_encoder_metadata(),
                    },
                )
            )
        return chunks, f"Docling extracted {len(sections)} structure-aware sections from {source_name}."
    except Exception as exc:
        return [], f"Docling extraction failed for {source_name}: {exc}"
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def _merge_units(units: list[str], mode: str) -> list[str]:
    if mode == "strict":
        return units
    if mode == "balanced":
        return _merge_by_size(units, target_chars=6000, max_chars=9000)
    return _merge_by_size(units, target_chars=10000, max_chars=14000)


def _merge_by_size(units: list[str], target_chars: int, max_chars: int) -> list[str]:
    sections: list[str] = []
    current: list[str] = []
    current_len = 0
    for unit in units:
        unit = unit.strip()
        if not unit:
            continue
        extra = len(unit) if not current else len(unit) + 2
        if current and current_len >= target_chars:
            sections.append("\n\n".join(current))
            current = [unit]
            current_len = len(unit)
            continue
        if current and current_len + extra > max_chars:
            sections.append("\n\n".join(current))
            current = [unit]
            current_len = len(unit)
            continue
        current.append(unit)
        current_len += extra
    if current:
        sections.append("\n\n".join(current))
    return sections

