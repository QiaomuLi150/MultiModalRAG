# Changelog

## Unreleased

### Added
- Optional visual page retrieval backed by ColQwen2-style multivector search.
- Hybrid retrieval modes with reranking and Light-ColPali-style compression.
- Guardrails with configurable levels.
- Evaluation runner, manifest-driven batch execution, and progress dashboard.
- Formal retrieval and answer-quality benchmark suites.
- Qdrant Cloud persistence and optional MUVERA-style proxy retrieval.
- Vision-first answer generation for visual modes.

### Changed
- The system is now focused on multimodal RAG only.
- KG/Neo4j functionality was removed from this repository and split out conceptually into a separate project.

### Fixed
- Video audio extraction now falls back to a bundled `ffmpeg` path when system `ffmpeg` is missing.
- OCR now uses a Python-only fallback path when the Tesseract binary is unavailable.
- Streamlit and CLI paths now reuse local secrets for API keys during evaluation and demo runs.
