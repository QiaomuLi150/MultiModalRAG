from __future__ import annotations

import base64
from io import BytesIO

from PIL import Image

from .generation import MODEL_NAME

VISION_PROMPT = """Describe this meeting asset for a multimodal RAG system.
Focus on visible slide text, objects, people, diagrams, chart trends, actions, and details useful for search.
Return 3-6 concise bullet points.
Do not invent details that are not visible."""


def describe_image_with_gpt5_nano(
    image: Image.Image,
    api_key: str | None,
    context: str = "",
) -> tuple[str, str]:
    if not api_key:
        return "", "OpenAI VLM skipped: no API key."

    try:
        from openai import OpenAI
    except Exception as exc:
        return "", f"OpenAI VLM unavailable: {exc}"

    prompt = VISION_PROMPT
    if context.strip():
        prompt += f"\n\nAdditional non-visual context:\n{context.strip()}"

    try:
        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=MODEL_NAME,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": _image_data_url(image)},
                    ],
                }
            ],
        )
        return response.output_text.strip(), f"OpenAI VLM completed with {MODEL_NAME}."
    except Exception as exc:
        return "", f"OpenAI VLM failed: {exc}"


def _image_data_url(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=85)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"

