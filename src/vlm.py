from __future__ import annotations

from PIL import Image

VLM_ENGINE = "qwen2.5-vl.optional"

VLM_PROMPT = """Describe this meeting-video segment for multimodal RAG.
Focus on visible slide text, diagrams, chart trends, people, actions, and details useful for search.
Return 3-6 concise bullets.
Do not invent details that are not visible."""


def describe_frame_with_local_vlm(image: Image.Image) -> tuple[str, str]:
    try:
        import transformers  # noqa: F401
    except Exception:
        return "", "VLM unavailable: install transformers plus a Qwen2.5-VL-compatible runtime."

    return (
        "",
        "VLM adapter placeholder: Qwen2.5-VL is intentionally optional for Streamlit Cloud. "
        "Install and wire a local model in src/vlm.py for automatic visual descriptions.",
    )

