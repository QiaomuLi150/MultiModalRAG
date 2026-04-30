from src.chunking import DocumentChunk
from src.citations import citation_label, format_evidence


def test_citation_label_format():
    chunk = DocumentChunk("chunk_001", "hello", "source.txt", "text")
    assert citation_label(chunk) == "[chunk_001]"


def test_evidence_includes_source_and_modality():
    chunk = DocumentChunk("chunk_002", "roadmap decision", "slide_notes.txt", "slide")
    evidence = format_evidence(chunk, score=0.42)
    assert "[chunk_002]" in evidence
    assert "source=slide_notes.txt" in evidence
    assert "modality=slide" in evidence
