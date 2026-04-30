from unittest.mock import MagicMock, patch

from src.audio import ASR_MODELS, audio_status_chunk, extract_audio_wav_from_video_bytes, transcribe_audio_bytes


def test_large_v3_is_available_as_asr_option():
    assert "large-v3" in ASR_MODELS


def test_video_audio_extraction_reports_missing_ffmpeg():
    with patch("subprocess.run", side_effect=FileNotFoundError):
        audio, status = extract_audio_wav_from_video_bytes(b"video", "meeting.mp4")

    assert audio == b""
    assert "ffmpeg" in status


def test_video_audio_extraction_reports_ffmpeg_failure():
    result = MagicMock()
    result.returncode = 1
    result.stderr = "no audio stream"

    with patch("subprocess.run", return_value=result):
        audio, status = extract_audio_wav_from_video_bytes(b"video", "meeting.mp4")

    assert audio == b""
    assert "failed" in status


def test_audio_status_chunk_is_searchable_fallback():
    chunk = audio_status_chunk("meeting.mp3", "ASR failed", "base")

    assert chunk.modality == "audio_transcript"
    assert chunk.metadata["chunker"] == "audio_status_fallback"
    assert "Audio asset uploaded" in chunk.text


def test_audio_transcription_retries_without_vad_when_empty(monkeypatch):
    calls = []

    class Segment:
        start = 0
        end = 1
        text = " hello"

    class Model:
        def transcribe(self, _path, **kwargs):
            calls.append(kwargs["vad_filter"])
            if kwargs["vad_filter"]:
                return iter([]), None
            return iter([Segment()]), None

    monkeypatch.setattr("src.audio._load_whisper_model", lambda _model_size: Model())
    chunks, status = transcribe_audio_bytes(b"fake audio", "meeting.wav")

    assert calls == [True, False]
    assert chunks[0].text == "hello"
    assert "retrying with VAD disabled" in status
