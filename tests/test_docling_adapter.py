from types import SimpleNamespace

from src.docling_adapter import _merge_units, chunks_from_pdf_docling_bytes


def test_docling_merge_balanced_combines_units():
    sections = _merge_units(["a" * 100, "b" * 100], mode="balanced")

    assert len(sections) == 1
    assert "a" in sections[0]
    assert "b" in sections[0]


def test_docling_extraction_reports_missing_dependency(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name.startswith("docling"):
            raise ImportError("missing docling")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    chunks, status = chunks_from_pdf_docling_bytes(b"%PDF fake", "slides.pdf")

    assert chunks == []
    assert "Docling unavailable" in status


def test_docling_extraction_builds_section_chunks(monkeypatch):
    class FakeConverter:
        def convert(self, source):
            return SimpleNamespace(document="doc")

    class FakeChunk:
        def __init__(self, text):
            self.text = text

    class FakeChunker:
        def chunk(self, dl_doc):
            return [FakeChunk("Roadmap section"), FakeChunk("Risk section")]

    monkeypatch.setitem(__import__("sys").modules, "docling.document_converter", SimpleNamespace(DocumentConverter=FakeConverter))
    monkeypatch.setitem(__import__("sys").modules, "docling.chunking", SimpleNamespace(HierarchicalChunker=FakeChunker))

    chunks, status = chunks_from_pdf_docling_bytes(b"%PDF fake", "slides.pdf")

    assert chunks
    assert chunks[0].metadata["converter"] == "docling_hierarchical_pdf"
    assert "Docling extracted" in status
