"""Provider router with failover logic and exponential backoff."""

import asyncio
import logging
import time

from nova.providers.base import (
    AllProvidersFailedError,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class ProviderRouter:
    """Routes requests through an ordered list of providers with automatic failover.

    Tries each provider in priority order. On transient failures (rate limits, timeouts),
    moves to the next provider. Implements exponential backoff for retries on the same
    provider across successive calls.
    """

    def __init__(self, provider_type: str, providers: list) -> None:
        """Initialize the router.

        Args:
            provider_type: Human-readable type name (e.g. "STT", "LLM", "TTS").
            providers: Ordered list of provider instances (highest priority first).
        """
        if not providers:
            raise ValueError(f"At least one {provider_type} provider is required")
        self.provider_type = provider_type
        self.providers = providers
        # Track backoff state per provider name: {name: (fail_count, last_fail_time)}
        self._backoff: dict[str, tuple[int, float]] = {}

    def _get_backoff_delay(self, provider_name: str) -> float:
        """Calculate current backoff delay for a provider.

        Returns 0 if no backoff is active or the backoff window has expired.
        Backoff schedule: 1s, 2s, 4s, 8s, 16s (capped).
        Backoff resets after 60 seconds of no failures.
        """
        if provider_name not in self._backoff:
            return 0.0
        fail_count, last_fail_time = self._backoff[provider_name]
        elapsed = time.monotonic() - last_fail_time
        # Reset backoff after 60s of no failures
        if elapsed > 60.0:
            del self._backoff[provider_name]
            return 0.0
        delay = min(2 ** (fail_count - 1), 16.0)
        remaining = delay - elapsed
        return max(remaining, 0.0)

    def _record_failure(self, provider_name: str) -> None:
        """Record a failure for backoff tracking."""
        if provider_name in self._backoff:
            fail_count, _ = self._backoff[provider_name]
            self._backoff[provider_name] = (fail_count + 1, time.monotonic())
        else:
            self._backoff[provider_name] = (1, time.monotonic())

    def _record_success(self, provider_name: str) -> None:
        """Reset backoff on success."""
        self._backoff.pop(provider_name, None)

    async def execute(self, method_name: str, *args, **kwargs):
        """Execute a method on providers with failover.

        Tries each provider in order. Skips providers that are in backoff.
        On RateLimitError or ProviderTimeoutError, records failure and tries next.
        On generic ProviderError, records failure and tries next.

        Args:
            method_name: The provider method to call (e.g. "transcribe", "generate").
            *args: Positional arguments to pass to the method.
            **kwargs: Keyword arguments to pass to the method.

        Returns:
            The result from the first successful provider.

        Raises:
            AllProvidersFailedError: When all providers have failed.
        """
        errors: list[ProviderError] = []

        for provider in self.providers:
            name = provider.name

            # Check backoff delay
            delay = self._get_backoff_delay(name)
            if delay > 0:
                logger.debug(
                    "[%s] %s is in backoff (%.1fs remaining), skipping",
                    self.provider_type, name, delay,
                )
                continue

            try:
                method = getattr(provider, method_name)
                logger.info("[%s] Trying %s", self.provider_type, name)
                result = await method(*args, **kwargs)
                self._record_success(name)
                logger.info("[%s] %s succeeded", self.provider_type, name)
                return result

            except RateLimitError as e:
                logger.warning(
                    "[%s] %s hit rate limit: %s", self.provider_type, name, e,
                )
                self._record_failure(name)
                errors.append(e)

            except ProviderTimeoutError as e:
                logger.warning(
                    "[%s] %s timed out: %s", self.provider_type, name, e,
                )
                self._record_failure(name)
                errors.append(e)

            except ProviderError as e:
                logger.warning(
                    "[%s] %s failed: %s", self.provider_type, name, e,
                )
                self._record_failure(name)
                errors.append(e)

        # All providers either failed or were in backoff — retry providers in backoff
        # with the shortest remaining delay (one retry pass)
        backoff_providers = [
            (self._get_backoff_delay(p.name), p)
            for p in self.providers
            if p.name not in {e.provider_name for e in errors}
        ]
        if backoff_providers:
            backoff_providers.sort(key=lambda x: x[0])
            delay, provider = backoff_providers[0]
            if delay > 0:
                logger.info(
                    "[%s] All providers exhausted, waiting %.1fs for %s",
                    self.provider_type, delay, provider.name,
                )
                await asyncio.sleep(delay)
            try:
                method = getattr(provider, method_name)
                logger.info("[%s] Retrying %s after backoff", self.provider_type, provider.name)
                result = await method(*args, **kwargs)
                self._record_success(provider.name)
                return result
            except (RateLimitError, ProviderTimeoutError, ProviderError) as e:
                self._record_failure(provider.name)
                errors.append(e)

        raise AllProvidersFailedError(self.provider_type, errors)
