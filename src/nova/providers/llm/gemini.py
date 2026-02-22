"""Gemini LLM provider — primary LLM using Google GenAI SDK with function calling."""

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

_BASE_SYSTEM_PROMPT = (
    "You are NOVA, a personal AI assistant created by Zhafran. "
    "You are modeled after JARVIS — calm, composed, and quietly competent. "
    "You speak with a refined, slightly formal tone but never stiff or robotic. "
    "You have subtle dry wit and occasionally make understated observations, "
    "but you never force humor or overdo it.\n\n"

    "Personality:\n"
    "- Address the user as 'Sir' or 'Pak' (in Indonesian) naturally, not every sentence.\n"
    "- Be efficient and precise — deliver information, not filler.\n"
    "- Show quiet confidence. You don't say 'I think' or 'maybe' — you state things.\n"
    "- When something goes wrong, stay composed: 'It appears the connection is unavailable' "
    "not 'Oh sorry I can't do that!'\n"
    "- Light sarcasm is acceptable when the user asks something obvious, but always respectful.\n"
    "- You are loyal and proactive — anticipate what the user might need next.\n\n"

    "Response rules:\n"
    "- Keep responses between 20-50 words. Only exceed if the user explicitly asks for detail.\n"
    "- Your responses will be spoken aloud — use plain spoken text only.\n"
    "- No markdown, bullet points, asterisks, or special characters.\n"
    "- No emoji. No exclamation marks unless truly warranted.\n"
    "- Default to Indonesian unless the user speaks in English.\n"
    "- Never say 'sebagai asisten' or reference your nature unless directly asked.\n"
    "- Never start with 'Tentu' or 'Baik' — just do or answer directly.\n\n"

    "Tool usage:\n"
    "- When the user asks you to perform an action, use the available tools immediately. "
    "Don't ask for confirmation unless the action is destructive (shutdown, restart, delete).\n"
    "- Only call web_search once per question. If results are insufficient, "
    "summarize what you found rather than searching again.\n"
    "- When web_search returns results, answer directly from them as if you knew the information. "
    "Never say 'saya menemukan hasil' or offer to open a browser.\n"
    "- When the user shares personal information, use remember_fact to store it.\n"
    "- When the user asks if you remember something, use recall_facts to check.\n"
)


def _build_system_prompt() -> str:
    """Build the full system prompt, injecting any stored user facts."""
    from nova.memory.persistent import get_user_memory

    facts = get_user_memory().get_facts()
    if not facts:
        return _BASE_SYSTEM_PROMPT

    facts_str = ", ".join(f"{k}={v}" for k, v in facts.items())
    return f"{_BASE_SYSTEM_PROMPT}\n\nKnown user facts: {facts_str}"

# Models in order of preference
_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash-lite"]

# Maximum number of function-call round-trips to prevent infinite loops
_MAX_TOOL_CALLS = 3


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

    def _get_config(
        self, tools: list | None = None,
    ) -> types.GenerateContentConfig:
        """Build the generation config with system instruction and optional tools.

        Args:
            tools: Optional list of Tool objects for function calling.

        Returns:
            GenerateContentConfig instance.
        """
        return types.GenerateContentConfig(
            system_instruction=_build_system_prompt(),
            tools=tools,
            temperature=0.3,
            max_output_tokens=150,
        )

    async def generate(
        self, prompt: str, context: list[dict], tools: list | None = None,
    ) -> str:
        """Generate a response, handling function calls if tools are provided.

        When tools are provided and the model returns function calls, this
        method executes them via the tool registry, feeds results back to the
        model, and returns the final text response.

        Args:
            prompt: The user's current message.
            context: Prior exchanges [{"role": "user"|"assistant", "content": str}].
            tools: Optional list of Tool objects for Gemini function calling.

        Returns:
            Generated response text.

        Raises:
            RateLimitError: On HTTP 429.
            ProviderError: On HTTP 500/503 or other API failures.
            ProviderTimeoutError: When the request exceeds llm_timeout.
        """
        client = self._get_client()
        contents = _build_contents(prompt, context)
        config = self._get_config(tools=tools)

        try:
            response = await client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )

            # Handle function calling loop
            if tools and response.function_calls:
                return await self._handle_function_calls(
                    client, contents, config, response,
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

    async def _handle_function_calls(
        self,
        client: genai.Client,
        contents: list[types.Content],
        config: types.GenerateContentConfig,
        response,
    ) -> str:
        """Execute function calls and continue the conversation.

        Enforces a per-tool timeout of 8 seconds. If a tool exceeds
        this, it returns an error to the model so it can respond with
        whatever information is available.

        Args:
            client: The GenAI client.
            contents: Conversation contents so far.
            config: Generation config with tools.
            response: The initial response containing function calls.

        Returns:
            Final text response after all function calls are resolved.
        """
        import asyncio

        from nova.tools.registry import execute_tool

        for iteration in range(_MAX_TOOL_CALLS):
            if not response.function_calls:
                break

            # Append the model's response (with function call parts)
            function_call_content = response.candidates[0].content
            contents.append(function_call_content)

            # Execute each function call and collect responses
            function_response_parts = []
            for fc_part in response.function_calls:
                fn_name = fc_part.name
                fn_args = dict(fc_part.args) if fc_part.args else {}

                logger.info(
                    "Gemini function call #%d: %s(%s)",
                    iteration + 1, fn_name, fn_args,
                )

                try:
                    result = await asyncio.wait_for(
                        execute_tool(fn_name, fn_args),
                        timeout=8.0,
                    )
                    fn_response = {"result": result}
                except TimeoutError:
                    logger.warning("Tool %s timed out (>8s)", fn_name)
                    fn_response = {"error": f"{fn_name} timed out"}
                except Exception as e:
                    logger.error("Tool %s failed: %s", fn_name, e)
                    fn_response = {"error": str(e)}

                function_response_parts.append(
                    types.Part.from_function_response(
                        name=fn_name,
                        response=fn_response,
                    )
                )

            # Add tool results to contents
            contents.append(
                types.Content(role="tool", parts=function_response_parts)
            )

            # Get next response from the model
            response = await client.aio.models.generate_content(
                model=self._model_name,
                contents=contents,
                config=config,
            )

        text = response.text
        if not text:
            raise ProviderError(
                self.name, "Gemini returned empty after tool calls",
            )
        logger.debug(
            "Gemini: generated %d chars (after tool calls)", len(text),
        )
        return text

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
