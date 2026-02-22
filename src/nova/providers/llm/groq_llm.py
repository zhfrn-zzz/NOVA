"""Groq LLM provider — fallback LLM using Groq API with Llama models."""

import json
import logging
from collections.abc import AsyncIterator

import httpx

from nova.config import get_config
from nova.providers.base import (
    LLMProvider,
    ProviderError,
    ProviderTimeoutError,
    RateLimitError,
)

logger = logging.getLogger(__name__)

_GROQ_CHAT_URL = "https://api.groq.com/openai/v1/chat/completions"
_MODEL = "llama-3.3-70b-versatile"

_BASE_SYSTEM_PROMPT = (
    "You are NOVA, a personal voice assistant. You run on a low-spec laptop "
    "and communicate via voice.\n\n"
    "Rules:\n"
    "- Keep responses under 100 words unless the user explicitly asks for detail.\n"
    "- Be conversational and natural — your responses will be spoken aloud.\n"
    "- Detect the user's language (Indonesian or English) and respond in the "
    "same language.\n"
    "- Don't use markdown formatting, bullet points, or special characters "
    "— plain spoken text only.\n"
    '- Don\'t say "as a voice assistant" or reference your nature unless asked.\n'
    "- Be helpful, direct, and friendly.\n"
    "- For questions you can't answer, say so briefly rather than making things up."
)


def _build_system_prompt() -> str:
    """Build the full system prompt, injecting any stored user facts."""
    from nova.memory.persistent import get_user_memory

    facts = get_user_memory().get_facts()
    if not facts:
        return _BASE_SYSTEM_PROMPT

    facts_str = ", ".join(f"{k}={v}" for k, v in facts.items())
    return f"{_BASE_SYSTEM_PROMPT}\n\nKnown user facts: {facts_str}"


def _build_messages(prompt: str, context: list[dict]) -> list[dict]:
    """Build the messages list for the chat completions API.

    Args:
        prompt: The user's current message.
        context: Prior exchanges [{"role": "user"|"assistant", "content": str}].

    Returns:
        List of message dicts including system prompt.
    """
    messages: list[dict] = [{"role": "system", "content": _build_system_prompt()}]
    for turn in context:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages


def _parse_retry_after_header(response: httpx.Response) -> float | None:
    """Extract retry-after seconds from response headers."""
    value = response.headers.get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return None


class GroqLLMProvider(LLMProvider):
    """LLM provider using Groq API with Llama models."""

    name = "groq_llm"

    def __init__(self) -> None:
        config = get_config()
        self._api_key = config.groq_api_key
        self._timeout = config.llm_timeout

    async def generate(
        self, prompt: str, context: list[dict], tools: list | None = None,
    ) -> str:
        """Generate a response given a prompt and conversation context.

        Args:
            prompt: The user's current message.
            context: Prior exchanges [{"role": "user"|"assistant", "content": str}].
            tools: Ignored — Groq fallback does not support function calling.

        Returns:
            Generated response text.

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On server or API errors.
            ProviderTimeoutError: When request exceeds llm_timeout.
        """
        if not self._api_key:
            raise ProviderError(self.name, "Groq API key not configured")

        messages = _build_messages(prompt, context)
        payload = {
            "model": _MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    _GROQ_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )

            if response.status_code == 429:
                retry_after = _parse_retry_after_header(response)
                raise RateLimitError(self.name, retry_after=retry_after)

            if response.status_code >= 500:
                raise ProviderError(
                    self.name,
                    f"Server error {response.status_code}: {response.text}",
                )

            if response.status_code != 200:
                raise ProviderError(
                    self.name,
                    f"API error {response.status_code}: {response.text}",
                )

            data = response.json()
            text = data["choices"][0]["message"]["content"].strip()

            if not text:
                raise ProviderError(self.name, "Groq returned empty response")

            logger.debug("Groq LLM: generated %d chars", len(text))
            return text

        except (RateLimitError, ProviderError, ProviderTimeoutError):
            raise
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(self.name, self._timeout) from e
        except Exception as e:
            raise ProviderError(self.name, f"Generation failed: {e}") from e

    async def generate_stream(
        self, prompt: str, context: list[dict],
    ) -> AsyncIterator[str]:
        """Stream a response token-by-token via Groq API.

        Args:
            prompt: The user's current message.
            context: Prior exchanges.

        Yields:
            Response text chunks as they arrive.
        """
        if not self._api_key:
            raise ProviderError(self.name, "Groq API key not configured")

        messages = _build_messages(prompt, context)
        payload = {
            "model": _MODEL,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 512,
            "stream": True,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                async with client.stream(
                    "POST",
                    _GROQ_CHAT_URL,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                ) as response:
                    if response.status_code == 429:
                        retry_after = _parse_retry_after_header(response)
                        raise RateLimitError(self.name, retry_after=retry_after)

                    if response.status_code != 200:
                        body = await response.aread()
                        raise ProviderError(
                            self.name,
                            f"API error {response.status_code}: {body.decode()}",
                        )

                    async for line in response.aiter_lines():
                        if not line.startswith("data: "):
                            continue
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            data = json.loads(data_str)
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                yield content
                        except (json.JSONDecodeError, KeyError, IndexError):
                            continue

        except (RateLimitError, ProviderError, ProviderTimeoutError):
            raise
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(self.name, self._timeout) from e
        except Exception as e:
            raise ProviderError(self.name, f"Stream failed: {e}") from e

    async def is_available(self) -> bool:
        """Check if the Groq API key is configured and valid."""
        if not self._api_key:
            return False
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {self._api_key}"},
                )
            return response.status_code == 200
        except Exception:
            logger.warning("Groq LLM: availability check failed")
            return False


if __name__ == "__main__":
    import asyncio
    import time

    async def main() -> None:
        provider = GroqLLMProvider()

        available = await provider.is_available()
        print(f"Groq LLM available: {available}")
        if not available:
            print("ERROR: Groq API key not configured or invalid.")
            return

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

        print("\n=== All tests complete ===")

    asyncio.run(main())
