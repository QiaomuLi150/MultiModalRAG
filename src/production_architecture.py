from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec


@dataclass(frozen=True)
class ProductionStage:
    stage: str
    target_component: str
    current_demo_component: str
    optional_packages: tuple[str, ...]
    output: str

    @property
    def availability(self) -> str:
        if not self.optional_packages:
            return "built in"
        installed = [package for package in self.optional_packages if find_spec(package) is not None]
        if len(installed) == len(self.optional_packages):
            return "optional packages installed"
        if installed:
            missing = ", ".join(package for package in self.optional_packages if package not in installed)
            return f"partially installed; missing {missing}"
        return "optional packages not installed"


def production_stages() -> list[ProductionStage]:
    return [
        ProductionStage(
            "Raw asset storage",
            "S3 or S3-compatible bucket",
            "Streamlit uploads and sample_data/",
            ("boto3",),
            "Stable asset URI plus checksum and file metadata",
        ),
        ProductionStage(
            "Audio transcription",
            "Whisper or faster-whisper",
            "Uploaded transcript text files",
            ("faster_whisper",),
            "Timestamped AUDIO transcript segments",
        ),
        ProductionStage(
            "Slide extraction",
            "Docling, Unstructured, or PaddleOCR",
            "PDF text, slide notes, CSV text, manual descriptions",
            ("docling",),
            "SLIDE text blocks with page/layout metadata",
        ),
        ProductionStage(
            "Video segmentation",
            "Time or scene segmentation with sampled frames",
            "OpenCV frame sampling behind a toggle",
            ("cv2",),
            "VIDEO frame or segment records with timestamps",
        ),
        ProductionStage(
            "Video/visual description",
            "Qwen2.5-VL or another open-source VLM",
            "Manual description plus local visual-stat encoder",
            ("transformers",),
            "Searchable segment descriptions",
        ),
        ProductionStage(
            "Shared text embedding",
            "Open-source text embeddings",
            "TF-IDF text encoder",
            ("sentence_transformers",),
            "One shared text vector space across AUDIO / SLIDE / VIDEO descriptions",
        ),
        ProductionStage(
            "Vector metadata store",
            "pgvector, Qdrant, Milvus, or OpenSearch",
            "In-memory TF-IDF matrix and chunk metadata",
            ("qdrant_client",),
            "Vectors plus source, modality, timestamp, page, and asset URI metadata",
        ),
        ProductionStage(
            "Cross-modal retrieval",
            "Top matches across AUDIO / SLIDE / VIDEO",
            "Top TF-IDF matches across current chunk modalities",
            (),
            "Ranked evidence set with scores and citations",
        ),
        ProductionStage(
            "Context packing",
            "Token-budget packer",
            "Local approximate token-budget packer",
            (),
            "Grounded context block bounded by answer-model budget",
        ),
        ProductionStage(
            "Grounded answer",
            "Answer model over retrieved context",
            "OpenAI gpt-5-nano only for chat, optional retrieval-only fallback",
            ("openai",),
            "Cited answer or evidence-only response",
        ),
    ]


def production_rows() -> list[dict[str, str]]:
    return [
        {
            "stage": stage.stage,
            "target component": stage.target_component,
            "current demo component": stage.current_demo_component,
            "availability": stage.availability,
            "output": stage.output,
        }
        for stage in production_stages()
    ]

