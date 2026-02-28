"""Tests for Gemini embedding integration — embedder, auto-embed, backfill."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nova.memory.memory_store import MemoryStore
from nova.memory.retriever import MemoryRetriever

# --- Fixtures ---


@pytest.fixture
def store(tmp_path):
    """Create a MemoryStore with a temporary database."""
    db = tmp_path / "test.db"
    s = MemoryStore(db_path=db)
    yield s
    s.close()


# --- GeminiEmbedder circuit breaker tests ---


class TestGeminiEmbedderCircuitBreaker:
    @pytest.mark.asyncio
    async def test_circuit_breaker_opens_after_threshold(self):
        """After 3 consecutive failures, embed returns None without API call."""
        from nova.memory.embeddings import GeminiEmbedder, reset_embedder

        reset_embedder()

        with patch("nova.memory.embeddings.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                gemini_api_key="test-key",
                embedding_model="gemini-embedding-001",
                embedding_dimensions=3072,
                embedding_circuit_breaker_threshold=3,
                embedding_circuit_breaker_cooldown=300,
            )

            embedder = GeminiEmbedder()
            # Mock the client to raise errors
            embedder._client = MagicMock()
            embedder._client.models.embed_content.side_effect = RuntimeError(
                "API error",
            )

            # First 3 calls should attempt API and fail
            for _i in range(3):
                result = await embedder.embed("test")
                assert result is None

            # After threshold, should return None immediately (disabled)
            assert embedder._consecutive_failures >= 3
            assert embedder._disabled_until > 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_resets_on_success(self):
        """Successful embed resets the failure counter."""
        from nova.memory.embeddings import GeminiEmbedder, reset_embedder

        reset_embedder()

        with patch("nova.memory.embeddings.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                gemini_api_key="test-key",
                embedding_model="gemini-embedding-001",
                embedding_dimensions=3072,
                embedding_circuit_breaker_threshold=3,
                embedding_circuit_breaker_cooldown=300,
            )

            embedder = GeminiEmbedder()

            # Mock successful response
            mock_result = MagicMock()
            mock_result.embeddings = [MagicMock(values=[0.1, 0.2, 0.3])]
            embedder._client = MagicMock()
            embedder._client.models.embed_content.return_value = mock_result

            result = await embedder.embed("test")
            assert result == [0.1, 0.2, 0.3]
            assert embedder._consecutive_failures == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_retries_after_cooldown(self):
        """After cooldown period expires, the embedder retries."""
        from nova.memory.embeddings import GeminiEmbedder, reset_embedder

        reset_embedder()

        with patch("nova.memory.embeddings.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                gemini_api_key="test-key",
                embedding_model="gemini-embedding-001",
                embedding_dimensions=3072,
                embedding_circuit_breaker_threshold=3,
                embedding_circuit_breaker_cooldown=1,  # 1 second cooldown
            )

            embedder = GeminiEmbedder()
            embedder._client = MagicMock()
            embedder._client.models.embed_content.side_effect = RuntimeError(
                "API error",
            )

            # Trigger circuit breaker
            for _ in range(3):
                await embedder.embed("test")

            # Simulate cooldown expiry
            embedder._disabled_until = time.monotonic() - 1

            # Now provide a success response
            mock_result = MagicMock()
            mock_result.embeddings = [MagicMock(values=[1.0, 2.0])]
            embedder._client.models.embed_content.side_effect = None
            embedder._client.models.embed_content.return_value = mock_result

            result = await embedder.embed("test")
            assert result == [1.0, 2.0]
            assert embedder._consecutive_failures == 0


# --- MemoryStore embedding tests ---


class TestStoreMemoryWithEmbedding:
    @pytest.mark.asyncio
    async def test_store_memory_auto_embeds(self, store):
        """When embedding_fn is set, store_memory schedules embedding."""
        mock_embed = AsyncMock(return_value=[0.1, 0.2, 0.3])
        store.set_embedding_fn(mock_embed)

        store.store_memory("color", "blue")

        # Give the background task time to complete
        await asyncio.sleep(0.1)

        results = store.get_memories_with_embeddings()
        assert len(results) == 1
        assert results[0]["key"] == "color"
        assert len(results[0]["embedding"]) == 3
        mock_embed.assert_called_once_with("blue")

    @pytest.mark.asyncio
    async def test_store_memory_no_embed_when_fn_none(self, store):
        """Without embedding_fn, no embedding is stored."""
        store.store_memory("test", "value")
        results = store.get_memories_with_embeddings()
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_store_memory_embed_failure_graceful(self, store):
        """Embedding failure doesn't break store_memory."""
        mock_embed = AsyncMock(side_effect=RuntimeError("API down"))
        store.set_embedding_fn(mock_embed)

        store.store_memory("test", "value")
        await asyncio.sleep(0.1)

        # Memory should still be stored, just without embedding
        assert store.get_memory("test") == "value"
        results = store.get_memories_with_embeddings()
        assert len(results) == 0


class TestBackfillEmbeddings:
    @pytest.mark.asyncio
    async def test_backfill_embeds_null_embeddings(self, store):
        """backfill_embeddings embeds memories without embeddings."""
        store.store_memory("a", "apple")
        store.store_memory("b", "banana")

        mock_embed = AsyncMock(return_value=[0.5, 0.5, 0.5])
        count = await store.backfill_embeddings(embedding_fn=mock_embed)

        assert count == 2
        assert mock_embed.call_count == 2
        results = store.get_memories_with_embeddings()
        assert len(results) == 2

    @pytest.mark.asyncio
    async def test_backfill_skips_already_embedded(self, store):
        """backfill_embeddings skips memories that already have embeddings."""
        store.store_memory("a", "apple")
        store.store_embedding("a", [1.0, 1.0])
        store.store_memory("b", "banana")

        mock_embed = AsyncMock(return_value=[0.5, 0.5])
        count = await store.backfill_embeddings(embedding_fn=mock_embed)

        assert count == 1  # Only "b" was backfilled
        mock_embed.assert_called_once_with("banana")

    @pytest.mark.asyncio
    async def test_backfill_returns_zero_no_fn(self, store):
        """backfill_embeddings returns 0 when no fn is available."""
        store.store_memory("a", "apple")
        count = await store.backfill_embeddings()
        assert count == 0

    @pytest.mark.asyncio
    async def test_backfill_partial_failure(self, store):
        """backfill_embeddings continues on individual failures."""
        store.store_memory("a", "apple")
        store.store_memory("b", "banana")

        call_count = 0

        async def flaky_embed(text):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Flaky API")
            return [0.5, 0.5]

        count = await store.backfill_embeddings(embedding_fn=flaky_embed)
        assert count == 1  # Only second one succeeded


# --- Retriever with embedding tests ---


class TestRetrieverWithEmbedding:
    @pytest.mark.asyncio
    async def test_retriever_uses_embedding_fn(self, store):
        """Retriever uses embedding_fn for vector search."""
        store.store_memory("pet", "cat named Milo")
        store.store_embedding("pet", [1.0, 0.0, 0.0])

        mock_embed = AsyncMock(return_value=[1.0, 0.0, 0.0])
        retriever = MemoryRetriever(
            memory_store=store, embedding_fn=mock_embed,
        )

        results = await retriever.search("cat")
        assert len(results) >= 1
        assert results[0]["vector_score"] > 0.0
        mock_embed.assert_called_once_with("cat")

    @pytest.mark.asyncio
    async def test_retriever_falls_back_without_embedder(self, store):
        """Retriever works with FTS5 only when no embedder."""
        store.store_memory("pet", "cat named Milo")

        retriever = MemoryRetriever(memory_store=store, embedding_fn=None)
        results = await retriever.search("cat")
        assert len(results) >= 1
        assert results[0]["vector_score"] == 0.0


# --- get_embedder singleton tests ---


class TestGetEmbedder:
    def test_get_embedder_returns_none_without_key(self):
        """get_embedder returns None when no API key is configured."""
        from nova.memory.embeddings import get_embedder, reset_embedder

        reset_embedder()

        with patch("nova.memory.embeddings.get_config") as mock_config:
            mock_config.return_value = MagicMock(gemini_api_key="")
            result = get_embedder()
            assert result is None

        reset_embedder()

    def test_get_embedder_returns_instance_with_key(self):
        """get_embedder returns GeminiEmbedder when API key is set."""
        from nova.memory.embeddings import (
            GeminiEmbedder,
            get_embedder,
            reset_embedder,
        )

        reset_embedder()

        with patch("nova.memory.embeddings.get_config") as mock_config:
            mock_config.return_value = MagicMock(
                gemini_api_key="test-key",
                embedding_model="gemini-embedding-001",
                embedding_dimensions=3072,
                embedding_circuit_breaker_threshold=3,
                embedding_circuit_breaker_cooldown=300,
            )
            result = get_embedder()
            assert isinstance(result, GeminiEmbedder)

        reset_embedder()
