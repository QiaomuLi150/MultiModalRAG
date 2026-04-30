from unittest.mock import MagicMock, patch

from src.ingest import chunks_from_pdf_bytes


def test_pdf_pages_become_pdf_chunks():
    fake_page = MagicMock()
    fake_page.extract_text.return_value = "Roadmap PDF page about offline mode."
    fake_reader = MagicMock()
    fake_reader.pages = [fake_page]

    with patch("src.ingest.PdfReader", return_value=fake_reader):
        chunks = chunks_from_pdf_bytes(b"%PDF fake", "roadmap.pdf")

    assert chunks
    assert chunks[0].modality == "pdf"
    assert chunks[0].source_name == "roadmap.pdf"
    assert chunks[0].page_or_frame == "page 1"
