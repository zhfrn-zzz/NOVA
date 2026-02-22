"""Tests for ProviderRouter failover and backoff logic."""

import pytest

from nova.providers.base import (
    AllProvidersFailedError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)
from nova.providers.router import ProviderRouter


# --- Mock Providers ---


class MockProvider:
    """A mock provider that can be configured to succeed or fail."""

    def __init__(self, name: str, result: str = "ok") -> None:
        self.name = name
        self._result = result
        self.call_count = 0

    async def do_work(self, *args, **kwargs) -> str:
        self.call_count += 1
        return self._result

    async def is_available(self) -> bool:
        return True


class MockFailProvider:
    """A mock provider that always raises a specific error."""

    def __init__(self, name: str, error: ProviderError) -> None:
        self.name = name
        self._error = error
        self.call_count = 0

    async def do_work(self, *args, **kwargs) -> str:
        self.call_count += 1
        raise self._error

    async def is_available(self) -> bool:
        return False


class MockFailThenSucceedProvider:
    """A mock provider that fails N times then succeeds."""

    def __init__(self, name: str, fail_count: int, error: ProviderError, result: str = "ok") -> None:
        self.name = name
        self._fail_count = fail_count
        self._error = error
        self._result = result
        self.call_count = 0

    async def do_work(self, *args, **kwargs) -> str:
        self.call_count += 1
        if self.call_count <= self._fail_count:
            raise self._error
        return self._result


# --- Tests ---


class TestProviderRouterInit:
    def test_requires_at_least_one_provider(self):
        with pytest.raises(ValueError, match="At least one"):
            ProviderRouter("LLM", [])

    def test_accepts_single_provider(self):
        router = ProviderRouter("LLM", [MockProvider("test")])
        assert router.provider_type == "LLM"
        assert len(router.providers) == 1


class TestPrimaryProviderSuccess:
    @pytest.mark.asyncio
    async def test_primary_succeeds_returns_result(self):
        primary = MockProvider("primary", result="hello")
        fallback = MockProvider("fallback", result="backup")
        router = ProviderRouter("LLM", [primary, fallback])

        result = await router.execute("do_work")

        assert result == "hello"
        assert primary.call_count == 1
        assert fallback.call_count == 0

    @pytest.mark.asyncio
    async def test_passes_args_to_provider(self):
        class ArgCapture:
            name = "capture"
            call_count = 0

            async def do_work(self, text, language="en"):
                self.captured = (text, language)
                return "done"

        provider = ArgCapture()
        router = ProviderRouter("TTS", [provider])

        await router.execute("do_work", "hello", language="id")

        assert provider.captured == ("hello", "id")


class TestFailoverBehavior:
    @pytest.mark.asyncio
    async def test_rate_limit_falls_back_to_next(self):
        primary = MockFailProvider(
            "primary", RateLimitError("primary", retry_after=5.0)
        )
        fallback = MockProvider("fallback", result="from-fallback")
        router = ProviderRouter("LLM", [primary, fallback])

        result = await router.execute("do_work")

        assert result == "from-fallback"
        assert primary.call_count == 1
        assert fallback.call_count == 1

    @pytest.mark.asyncio
    async def test_timeout_falls_back_to_next(self):
        primary = MockFailProvider(
            "primary", ProviderTimeoutError("primary", timeout=10.0)
        )
        fallback = MockProvider("fallback", result="from-fallback")
        router = ProviderRouter("STT", [primary, fallback])

        result = await router.execute("do_work")

        assert result == "from-fallback"

    @pytest.mark.asyncio
    async def test_generic_error_falls_back_to_next(self):
        primary = MockFailProvider(
            "primary", ProviderError("primary", "API error")
        )
        fallback = MockProvider("fallback", result="recovered")
        router = ProviderRouter("TTS", [primary, fallback])

        result = await router.execute("do_work")

        assert result == "recovered"


class TestAllProvidersFailed:
    @pytest.mark.asyncio
    async def test_all_fail_raises_all_providers_failed(self):
        p1 = MockFailProvider("p1", RateLimitError("p1"))
        p2 = MockFailProvider("p2", ProviderTimeoutError("p2", timeout=5.0))
        router = ProviderRouter("LLM", [p1, p2])

        with pytest.raises(AllProvidersFailedError) as exc_info:
            await router.execute("do_work")

        assert exc_info.value.provider_type == "LLM"
        assert len(exc_info.value.errors) == 2
        assert "p1" in str(exc_info.value)
        assert "p2" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_single_provider_fails_raises_error(self):
        provider = MockFailProvider("solo", ProviderError("solo", "dead"))
        router = ProviderRouter("STT", [provider])

        with pytest.raises(AllProvidersFailedError):
            await router.execute("do_work")


class TestBackoffBehavior:
    @pytest.mark.asyncio
    async def test_failed_provider_skipped_on_next_call(self):
        """After a failure, the provider should be in backoff on the next call."""
        primary = MockFailProvider("primary", RateLimitError("primary"))
        fallback = MockProvider("fallback", result="ok")
        router = ProviderRouter("LLM", [primary, fallback])

        # First call: primary fails, fallback succeeds
        await router.execute("do_work")
        assert primary.call_count == 1

        # Second call: primary should be skipped (in backoff), fallback used directly
        result = await router.execute("do_work")
        assert result == "ok"
        assert primary.call_count == 1  # not called again
        assert fallback.call_count == 2

    @pytest.mark.asyncio
    async def test_success_resets_backoff(self):
        """A successful call should reset backoff for that provider."""
        provider = MockFailThenSucceedProvider(
            "flaky", fail_count=1,
            error=RateLimitError("flaky"),
            result="recovered",
        )
        fallback = MockProvider("fallback", result="backup")
        router = ProviderRouter("LLM", [provider, fallback])

        # First call: flaky fails, fallback succeeds
        result1 = await router.execute("do_work")
        assert result1 == "backup"

        # Manually reset backoff to simulate time passing
        router._backoff.clear()

        # Second call: flaky succeeds now, backoff should be cleared
        result2 = await router.execute("do_work")
        assert result2 == "recovered"
        assert provider.name not in router._backoff


class TestExceptionHierarchy:
    def test_rate_limit_is_provider_error(self):
        e = RateLimitError("test")
        assert isinstance(e, ProviderError)

    def test_timeout_is_provider_error(self):
        e = ProviderTimeoutError("test", timeout=5.0)
        assert isinstance(e, ProviderError)

    def test_rate_limit_includes_retry_after(self):
        e = RateLimitError("test", retry_after=30.0)
        assert e.retry_after == 30.0
        assert "30.0s" in str(e)

    def test_rate_limit_without_retry_after(self):
        e = RateLimitError("test")
        assert e.retry_after is None
        assert "retry after" not in str(e)

    def test_timeout_includes_duration(self):
        e = ProviderTimeoutError("test", timeout=10.0)
        assert e.timeout == 10.0
        assert "10.0s" in str(e)

    def test_all_providers_failed_lists_names(self):
        errors = [
            RateLimitError("gemini"),
            ProviderTimeoutError("groq", timeout=5.0),
        ]
        e = AllProvidersFailedError("LLM", errors)
        assert "gemini" in str(e)
        assert "groq" in str(e)
        assert e.provider_type == "LLM"

    def test_provider_error_includes_name(self):
        e = ProviderError("gemini", "something broke")
        assert e.provider_name == "gemini"
        assert "[gemini]" in str(e)
