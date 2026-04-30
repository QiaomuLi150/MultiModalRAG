from PIL import Image

from src.vlm_openai import describe_image_with_gpt5_nano


def test_openai_vlm_skips_without_key():
    image = Image.new("RGB", (16, 16), color="white")

    text, status = describe_image_with_gpt5_nano(image, api_key=None)

    assert text == ""
    assert "no API key" in status
