from unittest.mock import MagicMock, patch

from src.generation import generate_stepback_question


def test_generate_stepback_question_uses_gpt5_nano():
    fake_response = MagicMock()
    fake_response.output_text = "What roadmap decisions were made?"
    fake_client = MagicMock()
    fake_client.responses.create.return_value = fake_response

    with patch("openai.OpenAI", return_value=fake_client):
        rewritten = generate_stepback_question("What did they decide about July?", "sk-test")

    assert rewritten == "What roadmap decisions were made?"
    assert fake_client.responses.create.call_args.kwargs["model"] == "gpt-5-nano"
