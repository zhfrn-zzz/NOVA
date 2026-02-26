"""Tests for MemoryRetriever — hybrid FTS5 + vector search."""

import pytest

from nova.memory.memory_store import MemoryStore
from nova.memory.retriever import MemoryRetriever


@pytest.fixture
def store(tmp_path):
    """Create a MemoryStore with a temporary database."""
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=db)
    yield s
    s.close()


@pytest.fixture
def retriever(store):
    """Create a retriever with no embedding function."""
    return MemoryRetriever(memory_store=store, embedding_fn=None)


class TestFTS5OnlySearch:
    @pytest.mark.asyncio
    async def test_search_finds_by_keyword(self, store, retriever):
        store.store_memory("hobby", "programming")
        results = await retriever.search("programming")
        assert len(results) >= 1
        assert results[0]["key"] == "hobby"

    @pytest.mark.asyncio
    async def test_search_empty_db(self, retriever):
        results = await retriever.search("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_search_respects_top_k(self, store, retriever):
        for i in range(10):
            store.store_memory(f"fact{i}", f"value about item {i}")
        results = await retriever.search("item")
        assert len(results) <= retriever.TOP_K


class TestHybridSearch:
    @pytest.mark.asyncio
    async def test_vector_search_combined(self, store):
        store.store_memory("color", "blue")
        store.store_embedding("color", [1.0, 0.0, 0.0])

        async def mock_embed(text):
            return [1.0, 0.0, 0.0]

        retriever = MemoryRetriever(
            memory_store=store, embedding_fn=mock_embed,
        )
        results = await retriever.search("blue")
        assert len(results) >= 1
        # Should have both keyword and vector scores
        assert results[0]["vector_score"] > 0

    @pytest.mark.asyncio
    async def test_vector_fallback_on_error(self, store):
        store.store_memory("test", "value")

        async def failing_embed(text):
            raise RuntimeError("API down")

        retriever = MemoryRetriever(
            memory_store=store, embedding_fn=failing_embed,
        )
        # Should still return FTS5 results
        results = await retriever.search("test")
        assert len(results) >= 1


class TestFormatting:
    @pytest.mark.asyncio
    async def test_format_for_prompt(self, store, retriever):
        store.store_memory("name", "Zhafran")
        store.store_memory("city", "Bekasi")
        results = await retriever.search("Zhafran")
        formatted = retriever.format_for_prompt(results)
        assert "name: Zhafran" in formatted

    def test_format_empty(self, retriever):
        assert retriever.format_for_prompt([]) == ""


class TestCosineSimilarity:
    def test_identical_vectors(self):
        sim = MemoryRetriever._cosine_similarity(
            [1.0, 0.0], [1.0, 0.0],
        )
        assert abs(sim - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        sim = MemoryRetriever._cosine_similarity(
            [1.0, 0.0], [0.0, 1.0],
        )
        assert abs(sim) < 1e-6

    def test_zero_vector(self):
        sim = MemoryRetriever._cosine_similarity(
            [0.0, 0.0], [1.0, 0.0],
        )
        assert sim == 0.0
