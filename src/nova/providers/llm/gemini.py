"""Gemini LLM provider — primary LLM using Google GenAI SDK with function calling."""

import logging
import re
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


def _build_system_prompt() -> str:
    """Build the full system prompt from file-based components.

    Uses PromptAssembler to read SOUL.md, RULES.md, USER.md and inject
    the current datetime. Memory context is injected separately when
    the retriever is available.
    """
    from nova.memory.prompt_assembler import get_prompt_assembler

    return get_prompt_assembler().build()

# Models in order of preference
_MODELS = ["gemini-2.5-flash", "gemini-2.0-flash-lite"]

# Maximum number of function-call round-trips to prevent infinite loops
_MAX_TOOL_CALLS = 3

# --- Sentence boundary detection for streaming ---

# Abbreviations that should NOT be treated as sentence endings
_STREAM_ABBREVIATIONS = {
    "dr", "mr", "mrs", "ms", "prof", "jr", "sr", "vs", "etc", "inc", "ltd",
    "dll", "dsb", "dkk", "spt", "yth", "no", "vol", "hal", "tel", "fax",
}

# Pattern: sentence-ending punctuation followed by whitespace
_SENTENCE_BREAK_RE = re.compile(r"[.!?]\s")


def _extract_sentence(buffer: str) -> tuple[str | None, str]:
    """Extract the first complete sentence from a token buffer.

    Returns (sentence, remaining_buffer) if a sentence boundary is found,
    or (None, buffer) if no complete sentence yet.
    """
    # Check for newline boundary
    nl_idx = buffer.find("\n")
    if nl_idx > 0:
        sentence = buffer[:nl_idx].strip()
        remaining = buffer[nl_idx + 1 :]
        if sentence and len(sentence) >= 8:
            return sentence, remaining

    # Check for punctuation + whitespace boundaries
    for match in _SENTENCE_BREAK_RE.finditer(buffer):
        end = match.start() + 1  # Include the punctuation character
        candidate = buffer[:end].strip()
        remaining = buffer[match.end() :]

        if not candidate or len(candidate) < 8:
            continue

        # Skip abbreviations (e.g. "Dr. ")
        if candidate[-1] == ".":
            stripped = candidate[:-1].strip()
            word_before = stripped.rsplit(None, 1)[-1].lower() if stripped else ""
            if word_before in _STREAM_ABBREVIATIONS:
                continue
            # Skip decimal numbers (e.g. "3.14 ")
            if len(candidate) >= 2 and candidate[-2].isdigit():
                continue

        return candidate, remaining

    return None, buffer


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
            max_output_tokens=512,
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

        Enforces a per-tool timeout of 15 seconds. If a tool exceeds
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
                        timeout=15.0,
                    )
                    fn_response = {"result": result}
                except TimeoutError:
                    logger.warning("Tool %s timed out (>15s)", fn_name)
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
        self, prompt: str, context: list[dict], tools: list | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response as complete sentences, with inline tool execution.

        Buffers incoming tokens and yields each complete sentence as
        soon as a sentence boundary is detected (. ! ? followed by
        whitespace, or newline). This enables the TTS pipeline to start
        speaking the first sentence before the full response is ready.

        When tools are provided and the model emits a function_call chunk,
        the stream pauses, the tool is executed, the result is appended
        to the conversation, and a **new** stream is started to get the
        model's response that incorporates the tool result.

        Args:
            prompt: The user's current message.
            context: Prior exchanges.
            tools: Optional list of Tool objects for Gemini function calling.

        Yields:
            Complete sentences as they are detected in the token stream.
        """
        import asyncio as _asyncio

        client = self._get_client()
        contents = _build_contents(prompt, context)
        config = self._get_config(tools=tools)
        buffer = ""
        tool_call_count = 0

        while tool_call_count <= _MAX_TOOL_CALLS:
            function_call = None

            try:
                stream = await client.aio.models.generate_content_stream(
                    model=self._model_name,
                    contents=contents,
                    config=config,
                )
                async for chunk in stream:
                    # Check for function call in chunk parts
                    if chunk.candidates and chunk.candidates[0].content.parts:
                        for part in chunk.candidates[0].content.parts:
                            if part.function_call:
                                function_call = part.function_call
                                break
                            if part.text:
                                buffer += part.text
                                # Extract and yield complete sentences
                                while True:
                                    sentence, buffer = _extract_sentence(buffer)
                                    if sentence is None:
                                        break
                                    yield sentence
                    elif chunk.text:
                        buffer += chunk.text
                        while True:
                            sentence, buffer = _extract_sentence(buffer)
                            if sentence is None:
                                break
                            yield sentence

                    if function_call:
                        break

            except (RateLimitError, ProviderTimeoutError, ProviderError):
                raise
            except Exception as e:
                self._handle_error(e)

            # If no function call, we're done streaming
            if not function_call:
                break

            # Execute the tool
            tool_call_count += 1
            fn_name = function_call.name
            fn_args = dict(function_call.args) if function_call.args else {}
            logger.info(
                "Stream function call #%d: %s(%s)",
                tool_call_count, fn_name, fn_args,
            )

            from nova.tools.registry import execute_tool

            try:
                result = await _asyncio.wait_for(
                    execute_tool(fn_name, fn_args), timeout=15.0,
                )
            except TimeoutError:
                result = f"Tool {fn_name} timed out after 15s"
            except Exception as e:
                result = f"Tool {fn_name} error: {e}"

            logger.info("Tool %s result: %s", fn_name, str(result)[:100])

            # Add function call + result to contents for next stream round
            contents.append(
                types.Content(
                    role="model",
                    parts=[types.Part(function_call=function_call)],
                )
            )
            contents.append(
                types.Content(
                    role="user",
                    parts=[types.Part(function_response=types.FunctionResponse(
                        name=fn_name,
                        response={"result": str(result)},
                    ))],
                )
            )
            # Loop continues — new stream will incorporate tool result

        # Yield any remaining text in the buffer
        if buffer.strip():
            yield buffer.strip()

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
