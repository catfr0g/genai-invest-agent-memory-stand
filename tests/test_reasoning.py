from types import SimpleNamespace

from fastapi.testclient import TestClient
from langchain_core.messages import AIMessage

import pytest

from app import api_server
from app.agent import runner


def test_extract_reasoning_from_additional_kwargs():
    message = AIMessage(content="answer", additional_kwargs={"reasoning": "think step"})
    assert runner._extract_reasoning(message) == "think step"


def test_aimessage_from_openai_choice_preserves_reasoning():
    message = runner._aimessage_from_openai_choice(
        {
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "4",
                "reasoning": "Thinking Process:\n1. Add numbers",
            },
        },
        usage={"completion_tokens_details": {"reasoning_tokens": 12}},
        model_name="qwen3.5-4b",
        response_id="chatcmpl-test",
    )
    assert runner._extract_reasoning(message) == "Thinking Process:\n1. Add numbers"
    assert runner._extract_reasoning_tokens(message) == 12


def test_extract_reasoning_tokens_from_response_metadata():
    message = AIMessage(
        content="answer",
        response_metadata={
            "token_usage": {"completion_tokens_details": {"reasoning_tokens": 42}}
        },
    )
    assert runner._extract_reasoning_tokens(message) == 42


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


def test_chat_completion_includes_reasoning_when_requested(monkeypatch):
    async def run_research(user_id, session_id, query, **kwargs):
        return {
            "final_report": "ok",
            "reasoning": "chain of thought",
            "reasoning_tokens": 17,
        }

    monkeypatch.setattr(api_server, "_resolve_user", lambda authorization: "1001")
    monkeypatch.setattr(api_server, "run_research", run_research)

    response = TestClient(api_server.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        json={
            "messages": [{"role": "user", "content": "test"}],
            "reasoning": True,
        },
    )

    body = response.json()
    assert response.status_code == 200
    assert body["choices"][0]["message"]["reasoning"] == "chain of thought"
    assert body["usage"]["completion_tokens_details"]["reasoning_tokens"] == 17
