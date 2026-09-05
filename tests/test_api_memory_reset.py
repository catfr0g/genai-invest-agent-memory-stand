import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import api_server
from app.apikeys import hash_key
from app.memory.errors import MemoryResetError


class FakeMemory:
    def __init__(self):
        self.clear_calls = 0

    def clear_all(self) -> dict[str, int]:
        self.clear_calls += 1
        return {"working_keys": 1, "dialog_sessions": 2}


class FakeApiKeys:
    def __init__(self, raw_key: str):
        self.key_hash = hash_key(raw_key)

    def find_by_hash(self, key_hash: str):
        if key_hash == self.key_hash:
            return type("ApiKeyRecord", (), {"user_id": "1001"})()
        return None


def test_reset_memory_authenticates_before_clearing(monkeypatch):
    memory = FakeMemory()
    resolved_headers: list[str | None] = []

    def resolve_user(authorization: str | None) -> str:
        resolved_headers.append(authorization)
        if authorization != "Bearer valid":
            raise HTTPException(status_code=401, detail="unauthorized")
        return "1001"

    monkeypatch.setattr(api_server, "_resolve_user", resolve_user)
    monkeypatch.setattr(api_server, "_memory", memory)

    assert api_server.reset_memory("Bearer valid") == {
        "status": "reset",
        "deleted": {"working_keys": 1, "dialog_sessions": 2},
    }
    assert memory.clear_calls == 1

    with pytest.raises(HTTPException) as exc_info:
        api_server.reset_memory("Bearer invalid")

    assert exc_info.value.status_code == 401
    assert memory.clear_calls == 1
    assert resolved_headers == ["Bearer valid", "Bearer invalid"]


def test_reset_memory_http_route_returns_deletion_counts(monkeypatch):
    memory = FakeMemory()
    monkeypatch.setattr(api_server, "_resolve_user", lambda authorization: "1001")
    monkeypatch.setattr(api_server, "_memory", memory)

    response = TestClient(api_server.app).post(
        "/v1/memory/reset",
        headers={"Authorization": "Bearer valid"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "status": "reset",
        "deleted": {"working_keys": 1, "dialog_sessions": 2},
    }
    assert memory.clear_calls == 1


def test_reset_memory_http_uses_real_api_key_auth(monkeypatch):
    memory = FakeMemory()
    raw_key = "sk-valid"
    monkeypatch.setattr(
        api_server,
        "_mongo",
        type("Mongo", (), {"api_keys": FakeApiKeys(raw_key)})(),
    )
    monkeypatch.setattr(api_server, "_memory", memory)
    client = TestClient(api_server.app)

    valid_response = client.post(
        "/v1/memory/reset",
        headers={"Authorization": f"Bearer {raw_key}"},
    )
    invalid_response = client.post(
        "/v1/memory/reset",
        headers={"Authorization": "Bearer sk-invalid"},
    )

    assert valid_response.status_code == 200
    assert invalid_response.status_code == 401
    assert memory.clear_calls == 1


def test_reset_memory_http_reports_partial_failure(monkeypatch):
    class FailingMemory:
        def clear_all(self):
            raise MemoryResetError({"working_keys": 1}, "semantic_memories")

    monkeypatch.setattr(api_server, "_resolve_user", lambda authorization: "1001")
    monkeypatch.setattr(api_server, "_memory", FailingMemory())

    response = TestClient(api_server.app).post(
        "/v1/memory/reset",
        headers={"Authorization": "Bearer valid"},
    )

    assert response.status_code == 503
    assert response.json() == {
        "detail": {
            "status": "reset_failed",
            "message": "Reset неполный; повторите его до начала тестового run.",
            "failed_at": "semantic_memories",
            "deleted": {"working_keys": 1},
        }
    }
