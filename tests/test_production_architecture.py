from src.production_architecture import production_rows


def test_production_rows_cover_demanded_stages():
    stages = {row["stage"] for row in production_rows()}

    assert "Raw asset storage" in stages
    assert "Audio transcription" in stages
    assert "Slide extraction" in stages
    assert "Video/visual description" in stages
    assert "Shared text embedding" in stages
    assert "Vector metadata store" in stages
    assert "Cross-modal retrieval" in stages
    assert "Context packing" in stages
    assert "Grounded answer" in stages
