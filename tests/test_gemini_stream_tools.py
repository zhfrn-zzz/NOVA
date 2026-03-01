"""Tests for unified streaming-with-tools in GeminiProvider.generate_stream().

Mocks the Gemini streaming API to verify:
- Pure text streaming yields sentences correctly
- Function calls mid-stream trigger tool execution and resume streaming
- Multiple sequential tool calls work
- Tool timeouts are handled gracefully
"""

import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nova.providers.llm.gemini import GeminiProvider, _extract_sentence


# --- Helpers for building mock chunks ---


def _make_text_chunk(text: str):
    """Create a mock streaming chunk with text content."""
    part = MagicMock()
    part.text = text
    part.function_call = None

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    chunk = MagicMock()
    chunk.candidates = [candidate]
    chunk.text = text
    return chunk


def _make_function_call_chunk(fn_name: str, fn_args: dict | None = None):
    """Create a mock streaming chunk with a function call."""
    fc = MagicMock()
    fc.name = fn_name
    fc.args = fn_args or {}
    fc.id = "mock-call-id"

    part = MagicMock()
    part.text = None
    part.function_call = fc

    content = MagicMock()
    content.parts = [part]

    candidate = MagicMock()
    candidate.content = content

    chunk = MagicMock()
    chunk.candidates = [candidate]
    chunk.text = None
    return chunk


def _mock_execute_tool_factory(return_values):
    """Create an async mock for execute_tool with given return values.

    Args:
        return_values: A single value, list of values, or exception to raise.
    """
    mock = AsyncMock()
    if isinstance(return_values, list):
        mock.side_effect = return_values
    elif isinstance(return_values, type) and issubclass(return_values, Exception):
        mock.side_effect = return_values("error")
    else:
        mock.return_value = return_values
    return mock


@pytest.fixture()
def gemini_provider():
    """Create a GeminiProvider with a mocked client."""
    with patch("nova.providers.llm.gemini.get_config") as mock_config:
        mock_config.return_value.gemini_api_key = "test-key"
        mock_config.return_value.llm_timeout = 30.0
        provider = GeminiProvider()
    # Replace the client with a mock
    provider._client = MagicMock()
    return provider


@pytest.fixture()
def mock_registry():
    """Mock the nova.tools.registry module to avoid importing psutil etc.

    Yields a mock module whose ``execute_tool`` can be configured per test.
    """
    mock_mod = MagicMock()
    mock_mod.execute_tool = AsyncMock()

    # Ensure the registry module is available in sys.modules
    # so the deferred `from nova.tools.registry import execute_tool` works
    old = sys.modules.get("nova.tools.registry")
    sys.modules["nova.tools.registry"] = mock_mod
    yield mock_mod
    if old is not None:
        sys.modules["nova.tools.registry"] = old
    else:
        sys.modules.pop("nova.tools.registry", None)


class TestExtractSentence:
    """Unit tests for the _extract_sentence helper."""

    def test_no_sentence_boundary(self):
        sentence, remaining = _extract_sentence("Hello world")
        assert sentence is None
        assert remaining == "Hello world"

    def test_period_boundary(self):
        sentence, remaining = _extract_sentence("Hello world. How are you")
        assert sentence == "Hello world."
        assert remaining == "How are you"

    def test_exclamation_boundary(self):
        sentence, remaining = _extract_sentence("Hello world! How are you")
        assert sentence == "Hello world!"
        assert remaining == "How are you"

    def test_short_fragment_skipped(self):
        sentence, remaining = _extract_sentence("Ok. Next")
        assert sentence is None
        assert remaining == "Ok. Next"

    def test_abbreviation_skipped(self):
        sentence, remaining = _extract_sentence("Dr. Budi is here")
        assert sentence is None
        assert remaining == "Dr. Budi is here"


class TestGenerateStreamPureText:
    """Test generate_stream() with text-only responses (no tool calls)."""

    @pytest.mark.asyncio
    async def test_yields_complete_sentences(self, gemini_provider):
        """Streaming text chunks should be buffered and yielded as sentences."""
        chunks = [
            _make_text_chunk("Halo, saya Nova. "),
            _make_text_chunk("Saya bisa membantu Anda."),
        ]

        async def mock_stream(*args, **kwargs):
            for c in chunks:
                yield c

        gemini_provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_stream(),
        )

        with patch("nova.providers.llm.gemini._build_system_prompt", return_value="sys"):
            sentences = []
            async for s in gemini_provider.generate_stream("test", context=[]):
                sentences.append(s)

        # Should yield at least the content (possibly merged into one or two pieces)
        full = " ".join(sentences)
        assert "Halo" in full
        assert "Nova" in full
        assert "membantu" in full

    @pytest.mark.asyncio
    async def test_remaining_buffer_yielded(self, gemini_provider):
        """Text without sentence boundary should be yielded at end of stream."""
        chunks = [_make_text_chunk("No period here")]

        async def mock_stream(*args, **kwargs):
            for c in chunks:
                yield c

        gemini_provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_stream(),
        )

        with patch("nova.providers.llm.gemini._build_system_prompt", return_value="sys"):
            sentences = []
            async for s in gemini_provider.generate_stream("test", context=[]):
                sentences.append(s)

        assert len(sentences) == 1
        assert sentences[0] == "No period here"


class TestGenerateStreamWithToolCall:
    """Test generate_stream() handling function calls mid-stream."""

    @pytest.mark.asyncio
    async def test_tool_call_executes_and_resumes(
        self, gemini_provider, mock_registry,
    ):
        """When model emits function_call, tool should execute and new stream should start."""
        # First stream: model emits a function call
        first_chunks = [_make_function_call_chunk("get_current_time")]
        # Second stream: model responds with text after tool result
        second_chunks = [_make_text_chunk("Sekarang pukul 10:00 WIB.")]

        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                for c in first_chunks:
                    yield c
            else:
                for c in second_chunks:
                    yield c

        gemini_provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=lambda *a, **kw: mock_stream(),
        )

        mock_registry.execute_tool = AsyncMock(return_value="10:00")

        with (
            patch("nova.providers.llm.gemini._build_system_prompt", return_value="sys"),
            patch.object(gemini_provider, "_get_config", return_value=MagicMock()),
        ):
            sentences = []
            async for s in gemini_provider.generate_stream(
                "jam berapa?", context=[], tools=["mock_tool"],
            ):
                sentences.append(s)

        # Tool should have been called
        mock_registry.execute_tool.assert_called_once_with("get_current_time", {})
        # Should yield the text from the second stream
        full = " ".join(sentences)
        assert "10:00" in full

    @pytest.mark.asyncio
    async def test_multiple_sequential_tool_calls(
        self, gemini_provider, mock_registry,
    ):
        """Model can make multiple tool calls in sequence."""
        # Stream 1: first tool call
        stream1 = [_make_function_call_chunk("get_current_time")]
        # Stream 2: second tool call
        stream2 = [_make_function_call_chunk("get_current_date")]
        # Stream 3: final text response
        stream3 = [_make_text_chunk("Sekarang hari Sabtu, pukul 10:00 WIB.")]

        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            streams = {1: stream1, 2: stream2, 3: stream3}
            for c in streams.get(call_count, stream3):
                yield c

        gemini_provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=lambda *a, **kw: mock_stream(),
        )

        mock_registry.execute_tool = AsyncMock(
            side_effect=["10:00", "Sabtu, 1 Maret 2026"],
        )

        with (
            patch("nova.providers.llm.gemini._build_system_prompt", return_value="sys"),
            patch.object(gemini_provider, "_get_config", return_value=MagicMock()),
        ):
            sentences = []
            async for s in gemini_provider.generate_stream(
                "jam dan tanggal?", context=[], tools=["mock_tool"],
            ):
                sentences.append(s)

        assert mock_registry.execute_tool.call_count == 2
        full = " ".join(sentences)
        assert "Sabtu" in full or "10:00" in full

    @pytest.mark.asyncio
    async def test_tool_timeout_handled(
        self, gemini_provider, mock_registry,
    ):
        """Tool that times out should send error to model, not crash."""
        # Stream 1: tool call
        stream1 = [_make_function_call_chunk("slow_tool")]
        # Stream 2: model responds after getting timeout error
        stream2 = [_make_text_chunk("Maaf, tool sedang tidak tersedia.")]

        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                for c in stream1:
                    yield c
            else:
                for c in stream2:
                    yield c

        gemini_provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=lambda *a, **kw: mock_stream(),
        )

        mock_registry.execute_tool = AsyncMock(
            side_effect=TimeoutError("Tool timed out"),
        )

        with (
            patch("nova.providers.llm.gemini._build_system_prompt", return_value="sys"),
            patch.object(gemini_provider, "_get_config", return_value=MagicMock()),
        ):
            sentences = []
            async for s in gemini_provider.generate_stream(
                "test", context=[], tools=["mock_tool"],
            ):
                sentences.append(s)

        full = " ".join(sentences)
        assert "tidak tersedia" in full or len(full) > 0

    @pytest.mark.asyncio
    async def test_tool_error_handled(
        self, gemini_provider, mock_registry,
    ):
        """Tool that raises an exception should send error to model."""
        # Stream 1: tool call
        stream1 = [_make_function_call_chunk("broken_tool")]
        # Stream 2: model responds after error
        stream2 = [_make_text_chunk("Terjadi kesalahan saat menjalankan tool.")]

        call_count = 0

        async def mock_stream(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                for c in stream1:
                    yield c
            else:
                for c in stream2:
                    yield c

        gemini_provider._client.aio.models.generate_content_stream = AsyncMock(
            side_effect=lambda *a, **kw: mock_stream(),
        )

        mock_registry.execute_tool = AsyncMock(
            side_effect=RuntimeError("Connection failed"),
        )

        with (
            patch("nova.providers.llm.gemini._build_system_prompt", return_value="sys"),
            patch.object(gemini_provider, "_get_config", return_value=MagicMock()),
        ):
            sentences = []
            async for s in gemini_provider.generate_stream(
                "test", context=[], tools=["mock_tool"],
            ):
                sentences.append(s)

        full = " ".join(sentences)
        assert len(full) > 0  # Model should still produce a response


class TestGenerateStreamNoTools:
    """Test that generate_stream works correctly when tools=None."""

    @pytest.mark.asyncio
    async def test_no_tools_still_streams(self, gemini_provider):
        """Without tools, generate_stream should work as pure text streaming."""
        chunks = [_make_text_chunk("Ini adalah respons tanpa tools.")]

        async def mock_stream(*args, **kwargs):
            for c in chunks:
                yield c

        gemini_provider._client.aio.models.generate_content_stream = AsyncMock(
            return_value=mock_stream(),
        )

        with patch("nova.providers.llm.gemini._build_system_prompt", return_value="sys"):
            sentences = []
            async for s in gemini_provider.generate_stream(
                "test", context=[], tools=None,
            ):
                sentences.append(s)

        full = " ".join(sentences)
        assert "tanpa tools" in full
