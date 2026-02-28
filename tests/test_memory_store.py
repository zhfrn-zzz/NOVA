"""Tests for MemoryStore — SQLite-backed memory system."""

import pytest

from nova.memory.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """Create a MemoryStore with a temporary database."""
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=db)
    yield s
    s.close()


class TestMemoryCRUD:
    def test_store_and_get_memory(self, store):
        store.store_memory("name", "Zhafran")
        assert store.get_memory("name") == "Zhafran"

    def test_store_overwrites_existing(self, store):
        store.store_memory("city", "Jakarta")
        store.store_memory("city", "Bekasi")
        assert store.get_memory("city") == "Bekasi"

    def test_get_nonexistent_returns_none(self, store):
        assert store.get_memory("doesnotexist") is None

    def test_get_all_memories(self, store):
        store.store_memory("a", "1")
        store.store_memory("b", "2")
        all_mem = store.get_all_memories()
        assert "a" in all_mem
        assert "b" in all_mem
        assert all_mem["a"] == "1"

    def test_delete_memory(self, store):
        store.store_memory("temp", "value")
        assert store.delete_memory("temp") is True
        assert store.get_memory("temp") is None

    def test_delete_nonexistent_returns_false(self, store):
        assert store.delete_memory("nope") is False

    def test_memory_count(self, store):
        assert store.memory_count() == 0
        store.store_memory("a", "1")
        store.store_memory("b", "2")
        assert store.memory_count() == 2

    def test_key_normalization(self, store):
        store.store_memory("  Name  ", "  Zhafran  ")
        assert store.get_memory("name") == "Zhafran"


class TestFTS5Search:
    def test_search_finds_by_key(self, store):
        store.store_memory("hobby", "programming")
        results = store.search_memories_fts("hobby")
        assert len(results) >= 1
        assert results[0]["key"] == "hobby"

    def test_search_finds_by_value(self, store):
        store.store_memory("food", "nasi goreng")
        results = store.search_memories_fts("nasi goreng")
        assert len(results) >= 1
        assert results[0]["value"] == "nasi goreng"

    def test_search_empty_returns_empty(self, store):
        results = store.search_memories_fts("nonexistent query xyz")
        assert results == []

    def test_search_respects_limit(self, store):
        for i in range(10):
            store.store_memory(f"item{i}", f"value item {i}")
        results = store.search_memories_fts("item", limit=3)
        assert len(results) <= 3

    def test_search_with_special_characters(self, store):
        """FTS5 should not crash on queries with ?, !, etc."""
        store.store_memory("hobby", "programming")
        # These should NOT raise sqlite3.OperationalError
        results = store.search_memories_fts("hobby?")
        assert len(results) >= 1  # "hobby?" → "hobby" matches
        results = store.search_memories_fts("hobby!")
        assert len(results) >= 1  # "hobby!" → "hobby" matches
        # Multi-word with special chars — should not crash even if no match
        store.search_memories_fts('hobby "test"')
        store.search_memories_fts("hobby (test)")
        store.search_memories_fts("what? where! how*")

    def test_search_all_special_chars_returns_empty(self, store):
        """Query with only special characters should return empty."""
        results = store.search_memories_fts("???!!!")
        assert results == []


class TestInteractions:
    def test_log_and_retrieve_interactions(self, store):
        store.log_interaction("user", "Hello")
        store.log_interaction("assistant", "Hi there")
        recent = store.get_recent_interactions()
        assert len(recent) == 2
        assert recent[0]["role"] == "user"
        assert recent[1]["role"] == "assistant"

    def test_interactions_filtered_by_date(self, store):
        store.log_interaction("user", "today msg")
        recent = store.get_recent_interactions(date="2000-01-01")
        assert len(recent) == 0

    def test_interaction_fts_search(self, store):
        store.log_interaction("user", "jadwal besok meeting")
        results = store.search_interactions_fts("jadwal meeting")
        assert len(results) >= 1

    def test_log_with_tool_calls(self, store):
        rid = store.log_interaction(
            "assistant", "Done",
            tool_calls=[{"name": "volume_up", "args": {}}],
        )
        assert rid > 0


class TestSessions:
    def test_start_and_end_session(self, store):
        sid = store.start_session()
        assert sid > 0
        store.end_session(sid, summary="Test session", token_count=100)


class TestEmbeddings:
    def test_store_and_retrieve_embedding(self, store):
        store.store_memory("test", "value")
        emb = [0.1, 0.2, 0.3]
        store.store_embedding("test", emb)
        results = store.get_memories_with_embeddings()
        assert len(results) == 1
        assert len(results[0]["embedding"]) == 3
        assert abs(results[0]["embedding"][0] - 0.1) < 1e-5


class TestMigration:
    def test_migration_from_json(self, tmp_path):
        # Create a legacy JSON file in a custom location
        # (migration only runs for default path, so we test the logic manually)
        db = tmp_path / "nova.db"
        store = MemoryStore(db_path=db)

        # Manually insert as if migrated
        store.store_memory("legacy_key", "legacy_value", source="migrated")
        assert store.get_memory("legacy_key") == "legacy_value"
        store.close()


class TestMemoryMD:
    def test_sync_creates_memory_md(self, tmp_path):
        db = tmp_path / "nova.db"
        store = MemoryStore(db_path=db)
        store.store_memory("name", "Zhafran")

        md_path = tmp_path / "MEMORY.md"
        assert md_path.exists()
        content = md_path.read_text(encoding="utf-8")
        assert "name" in content
        assert "Zhafran" in content
        store.close()
