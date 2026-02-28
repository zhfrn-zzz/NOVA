"""Gemini embedding integration — vector embeddings for NOVA's memory system.

Uses the google-genai SDK to generate 3072-dimensional embeddings via the
gemini-embedding-001 model. Includes a circuit breaker that disables the
embedder after consecutive failures to avoid cascading errors.
"""

import asyncio
import logging
import time

from google import genai

from nova.config import get_config

logger = logging.getLogger(__name__)

# Module-level singleton
_instance: "GeminiEmbedder | None" = None


class GeminiEmbedder:
    """Generates text embeddings using Gemini's embedding API.

    Features a circuit breaker that tracks consecutive failures and
    disables embedding calls after a threshold is reached. After a
    cooldown period, the embedder retries automatically.
    """

    def __init__(self) -> None:
        """Initialize the embedder with config and Gemini client."""
        config = get_config()
        self._client = genai.Client(api_key=config.gemini_api_key)
        self._model = config.embedding_model
        self._threshold = config.embedding_circuit_breaker_threshold
        self._cooldown = config.embedding_circuit_breaker_cooldown

        # Circuit breaker state
        self._consecutive_failures = 0
        self._disabled_until: float = 0.0

    async def embed(self, text: str) -> list[float] | None:
        """Generate an embedding vector for the given text.

        Args:
            text: The text to embed.

        Returns:
            List of floats (embedding vector), or None if the circuit
            breaker is open or the API call fails.
        """
        # Circuit breaker check
        if self._consecutive_failures >= self._threshold:
            now = time.monotonic()
            if now < self._disabled_until:
                return None
            # Cooldown expired — retry
            logger.info("Embedding circuit breaker: cooldown expired, retrying")
            self._consecutive_failures = 0

        try:
            result = await asyncio.to_thread(
                self._client.models.embed_content,
                model=self._model,
                contents=text,
            )
            self._consecutive_failures = 0
            return result.embeddings[0].values
        except Exception as e:
            self._consecutive_failures += 1
            if self._consecutive_failures >= self._threshold:
                self._disabled_until = time.monotonic() + self._cooldown
                logger.warning(
                    "Embedding circuit breaker OPEN after %d failures, "
                    "disabled for %ds: %s",
                    self._consecutive_failures, self._cooldown, e,
                )
            else:
                logger.warning(
                    "Embedding failed (%d/%d): %s",
                    self._consecutive_failures, self._threshold, e,
                )
            return None


def get_embedder() -> GeminiEmbedder | None:
    """Get the singleton GeminiEmbedder instance.

    Returns None if no Gemini API key is configured.

    Returns:
        The shared GeminiEmbedder, or None.
    """
    global _instance
    if _instance is None:
        config = get_config()
        if not config.gemini_api_key:
            return None
        _instance = GeminiEmbedder()
    return _instance


def reset_embedder() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
