from src.chunking import DocumentChunk
from src.graph import graph_records_from_chunks
from src.retrieval import qdrant_point_id


def test_graph_records_include_qdrant_point_id_and_relationship_fields():
    chunk = DocumentChunk(
        chunk_id="chunk_001",
        text="The roadmap decision was approved.",
        source_name="meeting.pdf",
        modality="slide",
        page_or_frame="page-1",
        metadata={
            "parent_id": "parent_abc",
            "converter": "pypdf",
            "chunker": "page",
            "encoder": "sentence-transformer",
        },
    )

    records = graph_records_from_chunks([chunk], qdrant_collection="multimodal_chunks")

    assert records == [
        {
            "asset": {
                "id": "meeting.pdf",
                "name": "meeting.pdf",
                "modality": "slide",
            },
            "parent": {
                "id": "parent_abc",
                "source_name": "meeting.pdf",
                "modality": "slide",
                "timestamp": None,
                "page_or_frame": "page-1",
            },
            "chunk": {
                "id": "chunk_001",
                "text": "The roadmap decision was approved.",
                "source_name": "meeting.pdf",
                "modality": "slide",
                "timestamp": None,
                "page_or_frame": "page-1",
                "qdrant_point_id": qdrant_point_id(chunk),
                "qdrant_collection": "multimodal_chunks",
                "converter": "pypdf",
                "chunker": "page",
                "encoder": "sentence-transformer",
            },
        }
    ]
