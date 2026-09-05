from types import SimpleNamespace

import pytest

from app import api_server
from app.agent import runner


@pytest.mark.parametrize("reasoning", [False, True])
def test_model_forwards_reasoning_as_enable_thinking(monkeypatch, reasoning):
    captured = {}

    def init_chat_model(model_name, **kwargs):
        captured["model_name"] = model_name
        captured["kwargs"] = kwargs
        return object()

    monkeypatch.setattr(runner, "init_chat_model", init_chat_model)
    settings = SimpleNamespace(
        openai_api_key="test-key",
        openai_base_url="http://model.test/v1",
        research_model="test-model",
        research_model_max_tokens=1024,
    )

    runner._model(settings, reasoning=reasoning)

    assert captured["kwargs"]["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": reasoning}
    }


def test_chat_request_reasoning_defaults_to_false():
    request = api_server.ChatCompletionRequest(
        messages=[api_server.ChatMessage(role="user", content="test")]
    )

    assert request.reasoning is False


def test_chat_request_accepts_reasoning_true():
    request = api_server.ChatCompletionRequest.model_validate(
        {
            "messages": [{"role": "user", "content": "test"}],
            "reasoning": True,
        }
    )

    assert request.reasoning is True
