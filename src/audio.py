from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import shutil
import subprocess
from tempfile import NamedTemporaryFile

from .chunking import DocumentChunk, split_text
from .encoders import text_encoder_metadata


ASR_ENGINE = "faster_whisper.optional"
DEFAULT_ASR_MODEL = "base"
ASR_MODELS = ("tiny", "base", "small", "medium", "large-v3")


def transcribe_audio_bytes(
    data: bytes,
    source_name: str,
    model_size: str = DEFAULT_ASR_MODEL,
) -> tuple[list[DocumentChunk], str]:
    try:
        _load_whisper_model(model_size)
    except Exception as exc:
        status = f"ASR unavailable for {source_name}: {exc}"
        return [audio_status_chunk(source_name, status, model_size)], status

    suffix = Path(source_name).suffix or ".wav"
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name

    try:
        model = _load_whisper_model(model_size)
        chunks = _transcribe_to_chunks(model, tmp_path, source_name, model_size, vad_filter=True)
        retried_without_vad = False
        if not chunks:
            retried_without_vad = True
            chunks = _transcribe_to_chunks(model, tmp_path, source_name, model_size, vad_filter=False)
        if not chunks:
            status = (
                f"ASR completed with faster-whisper {model_size}, but no speech text was detected "
                "after retrying with VAD disabled."
            )
            return [audio_status_chunk(source_name, status, model_size)], status
        retry_note = " after retrying with VAD disabled" if retried_without_vad else ""
        return chunks, f"ASR completed with faster-whisper {model_size}{retry_note}."
    except Exception as exc:
        status = f"ASR failed for {source_name}: {exc}"
        return [audio_status_chunk(source_name, status, model_size)], status
    finally:
        Path(tmp_path).unlink(missing_ok=True)


def transcribe_video_audio_bytes(
    data: bytes,
    source_name: str,
    model_size: str = DEFAULT_ASR_MODEL,
) -> tuple[list[DocumentChunk], str]:
    audio_bytes, extract_status = extract_audio_wav_from_video_bytes(data, source_name)
    if not audio_bytes:
        return [audio_status_chunk(source_name, extract_status, model_size)], extract_status
    chunks, asr_status = transcribe_audio_bytes(audio_bytes, f"{source_name}.audio.wav", model_size=model_size)
    if not chunks:
        return [audio_status_chunk(source_name, f"{extract_status} {asr_status}", model_size)], f"{extract_status} {asr_status}"
    return chunks, f"{extract_status} {asr_status}"


def audio_status_chunk(source_name: str, status: str, model_size: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id="",
        text=(
            f"Audio asset uploaded: {source_name}. {status} "
            "Upload a transcript file for retrieval if local ASR cannot run in this environment."
        ),
        source_name=source_name,
        modality="audio_transcript",
        metadata={
            "converter": ASR_ENGINE,
            "chunker": "audio_status_fallback",
            "asr_model": model_size,
            "asr_status": status,
            **text_encoder_metadata(),
        },
    )


def _transcribe_to_chunks(
    model,
    audio_path: str,
    source_name: str,
    model_size: str,
    vad_filter: bool,
) -> list[DocumentChunk]:
    segments, _info = model.transcribe(
        audio_path,
        beam_size=1,
        vad_filter=vad_filter,
        condition_on_previous_text=False,
    )
    chunks: list[DocumentChunk] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue
        for chunk in split_text(
            text,
            source_name=source_name,
            modality="audio_transcript",
            metadata={
                "converter": ASR_ENGINE,
                "chunker": "whisper_timestamped_segment",
                "asr_model": model_size,
                "vad_filter": vad_filter,
                "timestamp_start": _format_timestamp(segment.start),
                "timestamp_end": _format_timestamp(segment.end),
                **text_encoder_metadata(),
            },
        ):
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source_name=chunk.source_name,
                    modality=chunk.modality,
                    timestamp=f"{_format_timestamp(segment.start)}-{_format_timestamp(segment.end)}",
                    metadata=chunk.metadata,
                )
            )
    return chunks


def extract_audio_wav_from_video_bytes(data: bytes, source_name: str) -> tuple[bytes, str]:
    suffix = Path(source_name).suffix or ".mp4"
    with NamedTemporaryFile(suffix=suffix, delete=False) as video_tmp:
        video_tmp.write(data)
        video_path = video_tmp.name
    audio_path = ""
    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as audio_tmp:
            audio_path = audio_tmp.name
        ffmpeg_binary = _ffmpeg_binary()
        command = [
            ffmpeg_binary,
            "-y",
            "-i",
            video_path,
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            audio_path,
        ]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120, check=False)
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or "unknown ffmpeg error").strip().splitlines()[-1]
            return b"", f"Video audio extraction failed: {detail}"
        return Path(audio_path).read_bytes(), "Extracted video audio with ffmpeg."
    except FileNotFoundError:
        return b"", "Video audio extraction unavailable: install ffmpeg."
    except subprocess.TimeoutExpired:
        return b"", "Video audio extraction timed out."
    except Exception as exc:
        return b"", f"Video audio extraction failed: {exc}"
    finally:
        Path(video_path).unlink(missing_ok=True)
        if audio_path:
            Path(audio_path).unlink(missing_ok=True)


@lru_cache(maxsize=3)
def _load_whisper_model(model_size: str):
    from faster_whisper import WhisperModel

    if model_size not in ASR_MODELS:
        raise ValueError(f"Unsupported ASR model: {model_size}")
    return WhisperModel(model_size, device="cpu", compute_type="int8")


def _format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    minutes, second = divmod(total, 60)
    hour, minute = divmod(minutes, 60)
    if hour:
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"


def _ffmpeg_binary() -> str:
    system_ffmpeg = shutil.which("ffmpeg")
    if system_ffmpeg:
        return system_ffmpeg
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise FileNotFoundError("ffmpeg") from exc
