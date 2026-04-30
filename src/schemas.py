from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PageRecord:
    doc_id: str
    page_id: str
    page_num: int
    source_name: str
    source_path: str
    image_path: str
    source_type: str
    width: int
    height: int
    ocr_text: str = ""
    title_hint: str | None = None
    metadata: dict = field(default_factory=dict)

    def payload(self) -> dict:
        return {
            "page_id": self.page_id,
            "doc_id": self.doc_id,
            "page_num": self.page_num,
            "source_name": self.source_name,
            "source_path": self.source_path,
            "image_path": self.image_path,
            "source_type": self.source_type,
            "width": self.width,
            "height": self.height,
            "ocr_text": self.ocr_text,
            "title_hint": self.title_hint,
            "metadata": self.metadata,
        }

