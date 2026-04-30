from src.chunking import DocumentChunk
from src.context import estimate_tokens, pack_context, packed_context_block


def test_context_packer_respects_small_budget():
    chunk = DocumentChunk("chunk_001", "alpha " * 1000, "long.txt", "text")
    packed = pack_context([(chunk, 0.9)], token_budget=350, reserved_tokens=100)

    assert packed
    assert sum(item.estimated_tokens for item in packed) <= 350
    assert "trimmed" in packed[0].packed_text


def test_packed_context_includes_modality_metadata():
    chunk = DocumentChunk("chunk_002", "roadmap", "meeting.txt", "audio_transcript", timestamp="00:10")
    packed = pack_context([(chunk, 0.8)], token_budget=1000)
    block = packed_context_block(packed)

    assert "[chunk_002]" in block
    assert "modality=audio_transcript" in block
    assert "timestamp=00:10" in block


def test_estimate_tokens_is_never_zero_for_nonempty_text():
    assert estimate_tokens("a") == 1
