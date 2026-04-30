from __future__ import annotations

from PIL import Image
import numpy as np


OCR_ENGINE = "rapidocr_or_pytesseract.optional"


def extract_image_text(image: Image.Image) -> tuple[str, str]:
    try:
        from rapidocr_onnxruntime import RapidOCR

        engine = _rapidocr_engine()
        ocr_result, _elapsed = engine(np.array(image.convert("RGB")))
        lines = []
        for item in ocr_result or []:
            if len(item) >= 2 and isinstance(item[1], str):
                text = item[1].strip()
                if text:
                    lines.append(text)
        text = "\n".join(lines).strip()
        if text:
            return text, "OCR completed with rapidocr-onnxruntime."
    except Exception:
        pass

    try:
        import pytesseract

        text = pytesseract.image_to_string(image).strip()
    except Exception as exc:
        return "", f"OCR unavailable: {exc}"
    if not text:
        return "", "OCR found no readable text."
    return text, "OCR completed with pytesseract."


def _rapidocr_engine():
    from rapidocr_onnxruntime import RapidOCR

    return RapidOCR()
