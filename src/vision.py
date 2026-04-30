from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

from PIL import Image


def image_metadata_text(file_name: str, description: str) -> str:
    description = description.strip()
    if description:
        return description
    return f"Image asset uploaded: {file_name}. No visual description was provided."


@dataclass(frozen=True)
class VideoSegmentFrame:
    image: Image.Image
    frame: str
    timestamp: str
    segment_id: str
    segment_start: str
    segment_end: str


def sample_video_frames(
    video_bytes: bytes,
    max_frames: int = 5,
    source_name: str = "uploaded.mp4",
) -> list[VideoSegmentFrame]:
    try:
        import cv2
    except Exception as exc:
        raise RuntimeError("OpenCV is not available for video frame sampling.") from exc

    suffix = Path(source_name).suffix or ".mp4"
    with NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    capture = cv2.VideoCapture(tmp_path)
    try:
        if not capture.isOpened():
            raise RuntimeError("OpenCV could not open the uploaded video.")
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        if frame_count <= 0:
            raise RuntimeError("OpenCV could not determine the video frame count.")

        sample_count = max(1, min(max_frames, frame_count))
        positions = [int(i * (frame_count - 1) / max(1, sample_count - 1)) for i in range(sample_count)]
        frames = []
        duration_seconds = frame_count / fps if fps > 0 else 0
        segment_seconds = duration_seconds / sample_count if sample_count else 0
        for index, position in enumerate(positions, start=1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, position)
            ok, frame = capture.read()
            if not ok:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            seconds = position / fps if fps > 0 else 0
            start = max(0.0, seconds - segment_seconds / 2)
            end = min(duration_seconds, seconds + segment_seconds / 2) if duration_seconds else seconds
            frames.append(
                VideoSegmentFrame(
                    image=image,
                    frame=str(position),
                    timestamp=_format_timestamp(seconds),
                    segment_id=f"segment_{index:03d}",
                    segment_start=_format_timestamp(start),
                    segment_end=_format_timestamp(end),
                )
            )
        return frames
    finally:
        capture.release()
        Path(tmp_path).unlink(missing_ok=True)


def _format_timestamp(seconds: float) -> str:
    total = int(round(seconds))
    minutes, second = divmod(total, 60)
    hour, minute = divmod(minutes, 60)
    if hour:
        return f"{hour:02d}:{minute:02d}:{second:02d}"
    return f"{minute:02d}:{second:02d}"
