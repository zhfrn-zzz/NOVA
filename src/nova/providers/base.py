"""Abstract base classes for all NOVA providers and custom exceptions."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

# --- Exceptions ---


class ProviderError(Exception):
    """Base exception for all provider errors."""

    def __init__(self, provider_name: str, message: str) -> None:
        self.provider_name = provider_name
        super().__init__(f"[{provider_name}] {message}")


class RateLimitError(ProviderError):
    """Raised when a provider hits its rate limit (HTTP 429)."""

    def __init__(self, provider_name: str, retry_after: float | None = None) -> None:
        self.retry_after = retry_after
        msg = "Rate limit exceeded"
        if retry_after is not None:
            msg += f" (retry after {retry_after}s)"
        super().__init__(provider_name, msg)


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out."""

    def __init__(self, provider_name: str, timeout: float) -> None:
        self.timeout = timeout
        super().__init__(provider_name, f"Request timed out after {timeout}s")


class AllProvidersFailedError(Exception):
    """Raised when every provider in the chain has failed."""

    def __init__(self, provider_type: str, errors: list[ProviderError]) -> None:
        self.provider_type = provider_type
        self.errors = errors
        names = [e.provider_name for e in errors]
        super().__init__(
            f"All {provider_type} providers failed: {', '.join(names)}"
        )


# --- Abstract Base Classes ---


class STTProvider(ABC):
    """Abstract base class for Speech-to-Text providers."""

    name: str

    @abstractmethod
    async def transcribe(self, audio_bytes: bytes) -> str:
        """Convert audio bytes (WAV) to text.

        Args:
            audio_bytes: Raw WAV audio data (16kHz, mono, 16-bit PCM).

        Returns:
            Transcribed text string.

        Raises:
            ProviderError: On API or processing failure.
            RateLimitError: When rate limit is hit.
            ProviderTimeoutError: When request exceeds timeout.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""


class LLMProvider(ABC):
    """Abstract base class for Large Language Model providers."""

    name: str

    @abstractmethod
    async def generate(self, prompt: str, context: list[dict]) -> str:
        """Generate a response given a prompt and conversation context.

        Args:
            prompt: The user's current message.
            context: List of prior exchanges [{"role": "user"|"assistant", "content": str}].

        Returns:
            Generated response text.

        Raises:
            ProviderError: On API or processing failure.
            RateLimitError: When rate limit is hit.
            ProviderTimeoutError: When request exceeds timeout.
        """

    @abstractmethod
    async def generate_stream(self, prompt: str, context: list[dict]) -> AsyncIterator[str]:
        """Stream a response token-by-token.

        Args:
            prompt: The user's current message.
            context: List of prior exchanges.

        Yields:
            Response text chunks as they arrive.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""


class TTSProvider(ABC):
    """Abstract base class for Text-to-Speech providers."""

    name: str

    @abstractmethod
    async def synthesize(self, text: str, language: str = "id") -> bytes:
        """Convert text to spoken audio bytes.

        Args:
            text: Text to speak.
            language: Language code ("id" for Indonesian, "en" for English, "auto" for detect).

        Returns:
            Audio bytes (MP3 format).

        Raises:
            ProviderError: On API or processing failure.
            RateLimitError: When rate limit is hit.
            ProviderTimeoutError: When request exceeds timeout.
        """

    @abstractmethod
    async def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""
