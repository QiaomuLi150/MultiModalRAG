from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class LocalEncoding:
    encoder_name: str
    vector: list[float]
    summary: str


TEXT_ENCODER_NAME = "sklearn.tfidf.word_ngrams"
IMAGE_ENCODER_NAME = "local.pil_numpy.visual_stats.v1"


def text_encoder_metadata() -> dict:
    return {
        "encoder": TEXT_ENCODER_NAME,
        "embedding_space": "text_tfidf",
    }


class VisualFeatureEncoder:
    """Small deterministic image/frame encoder for deployment-friendly visual metadata.

    This is not a VLM and does not infer semantics. It encodes observable low-level
    visual properties so image and video assets have a modality-specific local
    encoder path without using OpenAI for extraction.
    """

    encoder_name = IMAGE_ENCODER_NAME

    def encode_bytes(self, data: bytes) -> LocalEncoding:
        image = Image.open(BytesIO(data)).convert("RGB")
        return self.encode_image(image)

    def encode_image(self, image: Image.Image) -> LocalEncoding:
        rgb = np.asarray(image.convert("RGB").resize((128, 128)), dtype=np.float32) / 255.0
        gray = rgb.mean(axis=2)
        channel_means = rgb.mean(axis=(0, 1))
        channel_stds = rgb.std(axis=(0, 1))
        brightness = float(gray.mean())
        contrast = float(gray.std())
        edge_density = _edge_density(gray)
        aspect_ratio = image.width / max(1, image.height)
        hist_features = _rgb_histogram(rgb, bins=4)

        vector = [
            float(aspect_ratio),
            brightness,
            contrast,
            edge_density,
            *[float(value) for value in channel_means],
            *[float(value) for value in channel_stds],
            *hist_features,
        ]
        summary = (
            f"Local visual encoder summary: orientation={_orientation(aspect_ratio)}; "
            f"dominant_color={_dominant_color(channel_means)}; "
            f"brightness={_bucket(brightness, 0.33, 0.66, 'dark', 'medium', 'bright')}; "
            f"contrast={_bucket(contrast, 0.12, 0.25, 'low', 'medium', 'high')}; "
            f"edge_density={_bucket(edge_density, 0.05, 0.12, 'low', 'medium', 'high')}."
        )
        return LocalEncoding(self.encoder_name, vector, summary)


def _rgb_histogram(rgb: np.ndarray, bins: int) -> list[float]:
    features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(rgb[:, :, channel], bins=bins, range=(0.0, 1.0), density=False)
        total = max(1, int(hist.sum()))
        features.extend((hist / total).astype(float).tolist())
    return features


def _edge_density(gray: np.ndarray) -> float:
    dx = np.abs(np.diff(gray, axis=1)).mean()
    dy = np.abs(np.diff(gray, axis=0)).mean()
    return float((dx + dy) / 2.0)


def _orientation(aspect_ratio: float) -> str:
    if aspect_ratio > 1.15:
        return "landscape"
    if aspect_ratio < 0.85:
        return "portrait"
    return "square"


def _dominant_color(channel_means: np.ndarray) -> str:
    red, green, blue = channel_means.tolist()
    if max(red, green, blue) < 0.2:
        return "dark"
    if min(red, green, blue) > 0.8:
        return "light"
    if abs(red - green) < 0.08 and abs(green - blue) < 0.08:
        return "gray"
    if red >= green and red >= blue:
        return "red" if green < 0.55 else "yellow"
    if green >= red and green >= blue:
        return "green" if blue < 0.55 else "cyan"
    return "blue"


def _bucket(value: float, low: float, high: float, low_label: str, mid_label: str, high_label: str) -> str:
    if value < low:
        return low_label
    if value < high:
        return mid_label
    return high_label

