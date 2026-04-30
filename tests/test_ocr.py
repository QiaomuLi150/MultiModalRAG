from unittest.mock import patch

from PIL import Image

from src.ocr import extract_image_text


def test_ocr_reports_missing_dependency():
    image = Image.new("RGB", (16, 16), color="white")

    with patch.dict("sys.modules", {"pytesseract": None}):
        text, status = extract_image_text(image)

    assert text == ""
    assert "OCR unavailable" in status


def test_ocr_returns_text_when_available():
    image = Image.new("RGB", (16, 16), color="white")

    with patch("pytesseract.image_to_string", return_value="Roadmap\n"):
        text, status = extract_image_text(image)

    assert text == "Roadmap"
    assert "completed" in status
