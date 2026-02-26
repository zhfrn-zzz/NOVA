"""Tests for ConversationManager — persistent sliding window with compaction."""

import pytest

from nova.memory.conversation import ConversationManager
from nova.memory.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """Create a MemoryStore with a temporary database."""
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=db)
    yield s
    s.close()


@pytest.fixture
def manager(store):
    """Create a ConversationManager with no LLM function."""
    return ConversationManager(memory_store=store, llm_fn=None)


class TestBasicOperations:
    def test_empty_context(self, manager):
        assert manager.get_context() == []
        assert manager.turn_count == 0

    @pytest.mark.asyncio
    async def test_add_turn(self, manager):
        await manager.add_turn("user", "Hello")
        assert manager.turn_count == 1
        ctx = manager.get_context()
        assert ctx[0] == {"role": "user", "content": "Hello"}

    @pytest.mark.asyncio
    async def test_add_exchange(self, manager):
        await manager.add_exchange("how are you", "I am well")
        assert manager.turn_count == 2
        ctx = manager.get_context()
        assert ctx[0]["role"] == "user"
        assert ctx[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_multiple_exchanges(self, manager):
        for i in range(5):
            await manager.add_exchange(f"msg{i}", f"resp{i}")
        assert manager.turn_count == 10

    def test_clear(self, manager):
        manager._history = [{"role": "user", "content": "test"}]
        manager.clear()
        assert manager.get_context() == []
        assert manager.turn_count == 0


class TestPersistence:
    @pytest.mark.asyncio
    async def test_interactions_persisted_to_db(self, store, manager):
        await manager.add_exchange("hello", "world")
        recent = store.get_recent_interactions()
        assert len(recent) == 2
        assert recent[0]["role"] == "user"
        assert recent[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_restart_recovery(self, store):
        # Simulate first session
        mgr1 = ConversationManager(memory_store=store, llm_fn=None)
        await mgr1.add_exchange("original msg", "original resp")

        # Simulate restart by creating new manager
        mgr2 = ConversationManager(memory_store=store, llm_fn=None)
        ctx = mgr2.get_context()
        assert len(ctx) >= 2
        assert ctx[0]["content"] == "original msg"


class TestCompaction:
    @pytest.mark.asyncio
    async def test_compaction_triggers_at_max_turns(self, store):
        manager = ConversationManager(
            memory_store=store, llm_fn=None,
        )
        # Fill to max (20 exchanges = 40 messages)
        for i in range(20):
            await manager.add_exchange(f"user msg {i}", f"assistant resp {i}")

        # After compaction: should have fewer than MAX_TURNS*2
        assert manager.turn_count < 40

    @pytest.mark.asyncio
    async def test_compaction_with_llm_fn(self, store):
        async def mock_llm(prompt: str) -> str:
            if "Summarize" in prompt:
                return "Ringkasan: percakapan tentang testing"
            if "Extract" in prompt:
                return "hobby: testing\nfood: nasi goreng"
            return "NONE"

        manager = ConversationManager(
            memory_store=store, llm_fn=mock_llm,
        )

        for i in range(20):
            await manager.add_exchange(f"msg {i}", f"resp {i}")

        # Should have summary bridge + recent turns
        ctx = manager.get_context()
        assert any("Ringkasan" in t["content"] for t in ctx)

        # Auto-extracted facts should be stored
        assert store.get_memory("hobby") == "testing"
        assert store.get_memory("food") == "nasi goreng"

    @pytest.mark.asyncio
    async def test_compaction_writes_daily_log(self, store, tmp_path):
        manager = ConversationManager(
            memory_store=store, llm_fn=None,
        )
        for i in range(20):
            await manager.add_exchange(f"msg {i}", f"resp {i}")

        daily_dir = store._db_path.parent / "daily"
        assert daily_dir.exists()
        log_files = list(daily_dir.glob("*.md"))
        assert len(log_files) >= 1


class TestSessions:
    def test_start_session(self, manager):
        manager.start_session()
        assert manager._session_id is not None

    def test_end_session(self, manager):
        manager.start_session()
        manager.end_session(summary="Test")
        assert manager._session_id is None
