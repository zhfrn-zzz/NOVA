"""Gemini LLM provider — primary LLM using Google GenAI SDK."""

import logging
from collections.abc import AsyncIterator

from google import genai
from google.genai import types

from nova.config import get_config
from nova.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are NOVA, a personal voice assistant. You run on a low-spec laptop "
    "and communicate via voice.\n\n"
    "Rules:\n"
    "- Keep responses under 100 words unless the user explicitly asks for detail.\n"
    "- Be conversational and natural — your responses will be spoken aloud.\n"
    "- Detect the user's language (Indonesian or English) and respond in the "
    "same language.\n"
    "- Don't use markdown formatting, bullet points, or special characters "
    "— plain spoken text only.\n"
    "- Don't say \"as a voice assistant\" or reference your nature unless asked.\n"
    "- Be helpful, direct, and friendly.\n"
    "- For questions you can't answer, say so briefly rather than making things up."
)

# Models in order of preference
_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash-lite"]


def _build_contents(
    prompt: str, context: list[dict],
) -> list[types.Content]:
    """Build the contents list from conversation context and current prompt.

    Args:
        prompt: The user's current message.
        context: Prior exchanges [{"role": "user"|"assistant", "content": str}].

    Returns:
        List of Content objects suitable for generate_content.
    """
    contents: list[types.Content] = []
    for turn in context:
        # Map "assistant" role to "model" for Gemini API
        role = "model" if turn["role"] == "assistant" else turn["role"]
        contents.append(
            types.Content(role=role, parts=[types.Part(text=turn["content"])])
        )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=prompt)])
    )
    return contents


class GeminiProvider(LLMProvider):
    """LLM provider using Google Gemini via the google-genai SDK."""

    name = "gemini"

    def __init__(self) -> None:
        config = get_config()
        self._api_key = config.gemini_api_key
        self._timeout = config.llm_timeout
        self._model_name: str = _MODELS[0]
        self._client: genai.Client | None = None

        if self._api_key:
            self._client = genai.Client(api_key=self._api_key, vertexai=False)

    def _get_client(self) -> genai.Client:
        """Get the GenAI client, raising if not configured."""
        if self._client is None:
            raise ProviderError(self.name, "Gemini API key not configured")
        return self._client

    def _get_config(self) -> types.GenerateContentConfig:
        """Build the generation config with system instruction."""
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
        )

    async def generate(self, prompt: str, context: list[dict]) -> str:
        """Generate a response given a prompt and conversation context.

        Args:
            prompt: The user's current message.
            context: Prior exchanges [{"role": "user"|"assistant", "content": str}].

        Returns:
            Generated response text.

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On HTTP 500/503 or other API failures.
            ProviderTimeoutError: When the request exceeds llm_timeout.
        """
        client = self._get_client()
        contents = _build_contents(prompt, context)

        try:
            response = await client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=self._get_config(),
            )
            text = response.text
            if not text:
                raise ProviderError(self.name, "Gemini returned empty response")
            logger.debug("Gemini: generated %d chars", len(text))
            return text

        except (RateLimitError, ProviderTimeoutError, ProviderError):
            raise
        except Exception as e:
            return self._handle_error(e)

    async def generate_stream(
        self, prompt: str, context: list[dict],
    ) -> AsyncIterator[str]:
        """Stream a response token-by-token.

        Args:
            prompt: The user's current message.
            context: Prior exchanges.

        Yields:
            Response text chunks as they arrive.
        """
        client = self._get_client()
        contents = _build_contents(prompt, context)

        try:
            async for chunk in client.aio.models.generate_content_stream(
                model=self._model_name,
                contents=contents,
                config=self._get_config(),
            ):
                if chunk.text:
                    yield chunk.text

        except (RateLimitError, ProviderTimeoutError, ProviderError):
            raise
        except Exception as e:
            self._handle_error(e)

    async def is_available(self) -> bool:
        """Check if the Gemini API key is configured and valid."""
        if not self._api_key or not self._client:
            return False
        try:
            # Minimal API call to verify key works
            await self._client.aio.models.get(model=self._model_name)
            return True
        except Exception:
            logger.warning("Gemini: API key validation failed")
            return False

    def _handle_error(self, exc: Exception) -> str:
        """Map SDK exceptions to NOVA provider errors.

        Never returns normally — always raises.
        """
        msg = str(exc).lower()

        # Rate limit (HTTP 429)
        if "429" in msg or "resource exhausted" in msg or "rate limit" in msg:
            retry_after = _parse_retry_after(str(exc))
            raise RateLimitError(self.name, retry_after=retry_after) from exc

        # Timeout
        if "timeout" in msg or "deadline" in msg:
            raise ProviderTimeoutError(self.name, self._timeout) from exc

        # Server errors (500/503)
        if "500" in msg or "503" in msg or "internal" in msg:
            raise ProviderError(self.name, f"Server error: {exc}") from exc

        raise ProviderError(self.name, f"Generation failed: {exc}") from exc


def _parse_retry_after(msg: str) -> float | None:
    """Try to extract retry-after seconds from an error message."""
    import re

    match = re.search(r"(\d+(?:\.\d+)?)\s*s", msg)
    if match:
        return float(match.group(1))
    return None


if __name__ == "__main__":
    import asyncio
    import time

    async def main() -> None:
        provider = GeminiProvider()

        # Test 1: Indonesian question
        print("\n=== Test 1: Indonesian Question ===")
        start = time.perf_counter()
        response = await provider.generate(
            "Siapa presiden pertama Indonesia?", context=[],
        )
        elapsed = time.perf_counter() - start
        print(f"Response: {response}")
        print(f"Latency: {elapsed:.2f}s")

        # Test 2: English question
        print("\n=== Test 2: English Question ===")
        start = time.perf_counter()
        response = await provider.generate(
            "What is the speed of light?", context=[],
        )
        elapsed = time.perf_counter() - start
        print(f"Response: {response}")
        print(f"Latency: {elapsed:.2f}s")

        # Test 3: Context test (3 exchanges)
        print("\n=== Test 3: Context Test (3 exchanges) ===")
        context: list[dict] = []

        exchanges = [
            "Nama saya Zhafran.",
            "Saya suka bermain gitar.",
            "Siapa nama saya dan apa hobi saya?",
        ]

        for msg in exchanges:
            print(f"\nUser: {msg}")
            start = time.perf_counter()
            response = await provider.generate(msg, context=context)
            elapsed = time.perf_counter() - start
            print(f"Nova: {response}")
            print(f"Latency: {elapsed:.2f}s")

            # Add to context for next turn
            context.append({"role": "user", "content": msg})
            context.append({"role": "assistant", "content": response})

        print("\n=== All tests complete ===")

    asyncio.run(main())
