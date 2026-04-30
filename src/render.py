from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import re

from PIL import Image

from .ocr import extract_image_text
from .schemas import PageRecord


def render_pdf_bytes(
    data: bytes,
    source_name: str,
    out_dir: Path,
    render_scale: float = 1.5,
    include_ocr: bool = True,
) -> list[PageRecord]:
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    document = fitz.open(stream=data, filetype="pdf")
    doc_id = stable_doc_id(source_name, data)
    doc_dir = out_dir / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)

    page_records: list[PageRecord] = []
    try:
        matrix = fitz.Matrix(render_scale, render_scale)
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            image_path = doc_dir / f"page_{page_index + 1:03d}.png"
            image_path.write_bytes(pix.tobytes("png"))
            image = Image.open(BytesIO(image_path.read_bytes())).convert("RGB")
            ocr_text, _ocr_status = extract_image_text(image) if include_ocr else ("", "OCR skipped.")
            page_records.append(
                PageRecord(
                    doc_id=doc_id,
                    page_id=f"{doc_id}_p{page_index + 1:03d}",
                    page_num=page_index + 1,
                    source_name=source_name,
                    source_path=source_name,
                    image_path=str(image_path),
                    source_type="pdf",
                    width=image.width,
                    height=image.height,
                    ocr_text=ocr_text,
                    metadata={
                        "sha256": sha256(data).hexdigest(),
                        "render_scale": render_scale,
                    },
                )
            )
    finally:
        document.close()
    return page_records


def ingest_image_bytes(
    data: bytes,
    source_name: str,
    out_dir: Path,
    include_ocr: bool = True,
    title_hint: str | None = None,
) -> PageRecord:
    out_dir.mkdir(parents=True, exist_ok=True)
    doc_id = stable_doc_id(source_name, data)
    image = Image.open(BytesIO(data)).convert("RGB")
    image_path = out_dir / f"{doc_id}.png"
    image.save(image_path, format="PNG")
    ocr_text, _ocr_status = extract_image_text(image) if include_ocr else ("", "OCR skipped.")
    return PageRecord(
        doc_id=doc_id,
        page_id=f"{doc_id}_p001",
        page_num=1,
        source_name=source_name,
        source_path=source_name,
        image_path=str(image_path),
        source_type="image",
        width=image.width,
        height=image.height,
        ocr_text=ocr_text,
        title_hint=title_hint,
        metadata={"sha256": sha256(data).hexdigest()},
    )


def ingest_pil_image(
    image: Image.Image,
    source_name: str,
    out_dir: Path,
    include_ocr: bool = True,
    page_num: int = 1,
    title_hint: str | None = None,
    source_type: str = "image",
    extra_metadata: dict | None = None,
) -> PageRecord:
    out_dir.mkdir(parents=True, exist_ok=True)
    buffer = BytesIO()
    normalized = image.convert("RGB")
    normalized.save(buffer, format="PNG")
    data = buffer.getvalue()
    doc_id = stable_doc_id(source_name, data)
    image_path = out_dir / f"{doc_id}_p{page_num:03d}.png"
    image_path.write_bytes(data)
    ocr_text, _ocr_status = extract_image_text(normalized) if include_ocr else ("", "OCR skipped.")
    metadata = {"sha256": sha256(data).hexdigest(), **(extra_metadata or {})}
    return PageRecord(
        doc_id=doc_id,
        page_id=f"{doc_id}_p{page_num:03d}",
        page_num=page_num,
        source_name=source_name,
        source_path=source_name,
        image_path=str(image_path),
        source_type=source_type,
        width=normalized.width,
        height=normalized.height,
        ocr_text=ocr_text,
        title_hint=title_hint,
        metadata=metadata,
    )


def stable_doc_id(source_name: str, data: bytes) -> str:
    stem = Path(source_name).stem.lower()
    safe = re.sub(r"[^a-z0-9]+", "_", stem).strip("_") or "document"
    digest = sha256(data).hexdigest()[:12]
    return f"{safe}_{digest}"

