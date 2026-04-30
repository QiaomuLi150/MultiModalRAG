from io import BytesIO

from PIL import Image

from src.encoders import IMAGE_ENCODER_NAME, TEXT_ENCODER_NAME, VisualFeatureEncoder
from src.ingest import chunk_from_image_bytes, chunks_from_text_bytes


def test_text_chunks_include_text_encoder_metadata():
    chunks = chunks_from_text_bytes(b"roadmap transcript", "notes.txt")
    assert chunks[0].metadata["chunker"] == "overlapping_character_text"
    assert chunks[0].metadata["encoder"] == TEXT_ENCODER_NAME


def test_image_chunk_uses_local_visual_encoder():
    image = Image.new("RGB", (32, 16), color=(255, 0, 0))
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    chunk = chunk_from_image_bytes(buffer.getvalue(), "red.png", "red test image")

    assert chunk.modality == "image"
    assert chunk.metadata["chunker"] == "single_image_asset"
    assert chunk.metadata["encoder"] == IMAGE_ENCODER_NAME
    assert chunk.metadata["visual_vector"]
    assert "Local visual encoder summary" in chunk.text


def test_visual_encoder_is_deterministic_for_same_image():
    image = Image.new("RGB", (24, 24), color=(0, 0, 255))
    encoder = VisualFeatureEncoder()

    first = encoder.encode_image(image)
    second = encoder.encode_image(image)

    assert first.vector == second.vector
    assert first.summary == second.summary
