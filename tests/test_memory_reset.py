from fnmatch import fnmatch
from types import SimpleNamespace

import pytest

from app.memory.errors import MemoryResetError
from app.memory.mongo import MongoMemoryStore
from app.memory.store import MemoryStore
from app.memory.working import WorkingMemoryStore


class FakeRedis:
    def __init__(self, keys: list[str]):
        self.keys = set(keys)
        self.scan_patterns: list[str] = []

    def scan_iter(self, *, match: str):
        self.scan_patterns.append(match)
        return iter(key for key in sorted(self.keys) if fnmatch(key, match))

    def delete(self, *keys: str) -> int:
        deleted = 0
        for key in keys:
            if key in self.keys:
                self.keys.remove(key)
                deleted += 1
        return deleted


class FakeCollection:
    def __init__(self, count: int, error: Exception | None = None):
        self.remaining = count
        self.error = error
        self.delete_filters: list[dict] = []

    def create_index(self, *args, **kwargs):
        return "fake-index"

    def delete_many(self, query: dict):
        self.delete_filters.append(query)
        if self.error:
            raise self.error
        deleted = self.remaining
        self.remaining = 0
        return SimpleNamespace(deleted_count=deleted)


class FakeDatabase:
    def __init__(self, collections: dict[str, FakeCollection]):
        self.collections = collections

    def __getitem__(self, name: str) -> FakeCollection:
        return self.collections[name]


class FakeMongoClient:
    def __init__(self, database: FakeDatabase):
        self.database = database

    def __getitem__(self, name: str) -> FakeDatabase:
        return self.database


def test_working_memory_clear_all_removes_only_working_keys():
    redis_client = FakeRedis(
        ["working:1001:attack", "working:1002:victim", "librechat:session"]
    )

    deleted = WorkingMemoryStore(redis_client).clear_all()

    assert deleted == 2
    assert redis_client.scan_patterns == ["working:*"]
    assert redis_client.keys == {"librechat:session"}


def test_mongo_clear_all_removes_memory_collections_but_not_api_keys():
    collections = {
        "dialog_sessions": FakeCollection(1),
        "episodic_memories": FakeCollection(2),
        "semantic_memories": FakeCollection(3),
        "agent_policy_memories": FakeCollection(4),
        "api_keys": FakeCollection(5),
    }
    database = FakeDatabase(collections)
    store = MongoMemoryStore(FakeMongoClient(database))

    assert store.clear_all() == {
        "dialog_sessions": 1,
        "episodic_memories": 2,
        "semantic_memories": 3,
        "agent_policy_memories": 4,
    }
    for name in (
        "dialog_sessions",
        "episodic_memories",
        "semantic_memories",
        "agent_policy_memories",
    ):
        assert collections[name].delete_filters == [{}]
    assert collections["api_keys"].delete_filters == []


def test_memory_store_clear_all_is_idempotent_and_reports_all_backends():
    redis_client = FakeRedis(["working:1001:attack"])
    collections = {
        "dialog_sessions": FakeCollection(1),
        "episodic_memories": FakeCollection(0),
        "semantic_memories": FakeCollection(2),
        "agent_policy_memories": FakeCollection(1),
        "api_keys": FakeCollection(1),
    }
    mongo_store = MongoMemoryStore(FakeMongoClient(FakeDatabase(collections)))

    store = MemoryStore.__new__(MemoryStore)
    store.working = WorkingMemoryStore(redis_client)
    store.mongo = mongo_store

    expected = {
        "working_keys": 1,
        "dialog_sessions": 1,
        "episodic_memories": 0,
        "semantic_memories": 2,
        "agent_policy_memories": 1,
    }
    assert store.clear_all() == expected
    assert store.clear_all() == {
        "working_keys": 0,
        "dialog_sessions": 0,
        "episodic_memories": 0,
        "semantic_memories": 0,
        "agent_policy_memories": 0,
    }


def test_memory_store_clear_all_reports_partial_failure():
    redis_client = FakeRedis(["working:1001:attack"])
    collections = {
        "dialog_sessions": FakeCollection(1),
        "episodic_memories": FakeCollection(2),
        "semantic_memories": FakeCollection(3, error=RuntimeError("mongo unavailable")),
        "agent_policy_memories": FakeCollection(4),
        "api_keys": FakeCollection(5),
    }
    store = MemoryStore.__new__(MemoryStore)
    store.working = WorkingMemoryStore(redis_client)
    store.mongo = MongoMemoryStore(FakeMongoClient(FakeDatabase(collections)))

    with pytest.raises(MemoryResetError) as exc_info:
        store.clear_all()

    assert exc_info.value.failed_at == "semantic_memories"
    assert exc_info.value.deleted == {
        "working_keys": 1,
        "dialog_sessions": 1,
        "episodic_memories": 2,
    }
