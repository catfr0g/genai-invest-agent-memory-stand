from fastapi.testclient import TestClient

from app import api_server
from app.agent.runner import _memory_text


def _multimodal_content(image_count: int = 1) -> list[dict]:
    content = [{"type": "text", "text": "Что на изображении?"}]
    content.extend(
        {
            "type": "image_url",
            "image_url": {"url": f"data:image/png;base64,image-{index}"},
        }
        for index in range(image_count)
    )
    return content


def test_multimodal_content_is_forwarded_to_research_model(monkeypatch):
    captured = {}

    async def run_research(user_id, session_id, query, **kwargs):
        captured.update(
            user_id=user_id,
            session_id=session_id,
            query=query,
            kwargs=kwargs,
        )
        return {"final_report": "На изображении тест."}

    monkeypatch.setattr(api_server, "_resolve_user", lambda authorization: "1001")
    monkeypatch.setattr(api_server, "run_research", run_research)
    content = _multimodal_content()

    response = TestClient(api_server.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        json={"messages": [{"role": "user", "content": content}]},
    )

    assert response.status_code == 200
    assert captured["query"] == content


def test_more_than_one_image_is_rejected(monkeypatch):
    monkeypatch.setattr(api_server, "_resolve_user", lambda authorization: "1001")

    response = TestClient(api_server.app).post(
        "/v1/chat/completions",
        headers={"Authorization": "Bearer test"},
        json={"messages": [{"role": "user", "content": _multimodal_content(2)}]},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Поддерживается не более одного изображения"


def test_memory_text_does_not_store_image_data():
    content = _multimodal_content()

    text = _memory_text(content)

    assert text == "Что на изображении?\n[Изображение]"
    assert "base64" not in text
