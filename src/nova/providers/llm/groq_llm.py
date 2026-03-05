"""Groq LLM provider — fallback LLM using Groq API with Llama models.

Supports function calling via the OpenAI-compatible tool use API.
"""

import asyncio
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
_MAX_TOOL_CALLS = 3


def _build_system_prompt() -> str:
    """Build the full system prompt from file-based components.

    Uses PromptAssembler to read SOUL.md, RULES.md, USER.md and inject
    the current datetime. Same prompt source as the Gemini provider.
    """
    from nova.memory.prompt_assembler import get_prompt_assembler

    return get_prompt_assembler().build()


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


def _get_openai_tools() -> list[dict]:
    """Get tool declarations in OpenAI format (cached)."""
    from nova.tools.registry import get_tool_declarations_openai

    return get_tool_declarations_openai()


def _parse_retry_after_header(response: httpx.Response) -> float | None:
    """Extract retry-after seconds from response headers."""
    value = response.headers.get("retry-after")
    if value:
        try:
            return float(value)
        except ValueError:
            pass
    return None


def _check_response_status(name: str, response: httpx.Response) -> None:
    """Raise appropriate errors for non-200 HTTP responses."""
    if response.status_code == 429:
        retry_after = _parse_retry_after_header(response)
        raise RateLimitError(name, retry_after=retry_after)
    if response.status_code >= 500:
        raise ProviderError(name, f"Server error {response.status_code}: {response.text}")
    if response.status_code != 200:
        raise ProviderError(name, f"API error {response.status_code}: {response.text}")


async def _execute_tool_calls(tool_calls: list[dict]) -> list[dict]:
    """Execute tool calls and return tool result messages.

    Args:
        tool_calls: List of tool call dicts from the API response.

    Returns:
        List of message dicts with role="tool" and tool results.
    """
    from nova.tools.registry import execute_tool

    results = []
    for tc in tool_calls:
        tc_id = tc["id"]
        fn = tc["function"]
        fn_name = fn["name"]
        try:
            fn_args = json.loads(fn["arguments"]) if fn["arguments"] else {}
        except json.JSONDecodeError:
            fn_args = {}

        logger.info("Groq tool call: %s(%s)", fn_name, fn_args)

        from nova.tools.registry import get_tool_timeout

        timeout = get_tool_timeout(fn_name)
        try:
            result = await asyncio.wait_for(
                execute_tool(fn_name, fn_args), timeout=timeout,
            )
        except TimeoutError:
            result = f"Tool {fn_name} timed out after {timeout:.0f}s"
        except Exception as e:
            result = f"Tool {fn_name} error: {e}"

        logger.info("Tool %s result: %s", fn_name, str(result)[:100])
        results.append({
            "role": "tool",
            "tool_call_id": tc_id,
            "content": str(result),
        })
    return results


class GroqLLMProvider(LLMProvider):
    """LLM provider using Groq API with Llama models.

    Supports function calling via OpenAI-compatible tool use API.
    """

    name = "groq_llm"

    def __init__(self) -> None:
        config = get_config()
        self._api_key = config.groq_api_key
        self._timeout = config.llm_timeout

    def _headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def generate(
        self, prompt: str, context: list[dict], tools: list | None = None,
    ) -> str:
        """Generate a response with optional function calling.

        Args:
            prompt: The user's current message.
            context: Prior exchanges [{"role": "user"|"assistant", "content": str}].
            tools: Optional tool declarations (Gemini format, ignored — uses
                   OpenAI format from registry directly).

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
        openai_tools = _get_openai_tools() if tools else None

        try:
            tool_call_count = 0
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                while tool_call_count <= _MAX_TOOL_CALLS:
                    payload: dict = {
                        "model": _MODEL,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 512,
                    }
                    if openai_tools:
                        payload["tools"] = openai_tools
                        payload["tool_choice"] = "auto"

                    response = await client.post(
                        _GROQ_CHAT_URL, headers=self._headers(), json=payload,
                    )
                    _check_response_status(self.name, response)

                    data = response.json()
                    message = data["choices"][0]["message"]

                    # Check for tool calls
                    if message.get("tool_calls"):
                        tool_call_count += 1
                        tool_calls = message["tool_calls"]
                        logger.info(
                            "Groq function call round #%d: %d tool(s)",
                            tool_call_count, len(tool_calls),
                        )

                        # Add assistant message with tool_calls to history
                        messages.append({
                            "role": "assistant",
                            "content": message.get("content") or "",
                            "tool_calls": tool_calls,
                        })

                        # Execute tools and add results
                        tool_results = await _execute_tool_calls(tool_calls)
                        messages.extend(tool_results)
                        continue

                    # No tool calls — return text response
                    text = (message.get("content") or "").strip()
                    if not text:
                        raise ProviderError(self.name, "Groq returned empty response")

                    logger.debug("Groq LLM: generated %d chars", len(text))
                    return text

            # Exhausted tool call rounds
            raise ProviderError(self.name, f"Exceeded {_MAX_TOOL_CALLS} tool call rounds")

        except (RateLimitError, ProviderError, ProviderTimeoutError):
            raise
        except httpx.TimeoutException as e:
            raise ProviderTimeoutError(self.name, self._timeout) from e
        except Exception as e:
            raise ProviderError(self.name, f"Generation failed: {e}") from e

    async def generate_stream(
        self, prompt: str, context: list[dict], tools: list | None = None,
    ) -> AsyncIterator[str]:
        """Stream a response with inline function calling via Groq API.

        When tools are provided and the model emits tool_calls, the stream
        pauses, tools are executed, results are added to messages, and a new
        stream is started — same pattern as the Gemini streaming provider.

        Args:
            prompt: The user's current message.
            context: Prior exchanges.
            tools: Optional tool declarations (triggers OpenAI tool use).

        Yields:
            Response text chunks as they arrive.
        """
        if not self._api_key:
            raise ProviderError(self.name, "Groq API key not configured")

        messages = _build_messages(prompt, context)
        openai_tools = _get_openai_tools() if tools else None
        tool_call_count = 0

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                while tool_call_count <= _MAX_TOOL_CALLS:
                    payload: dict = {
                        "model": _MODEL,
                        "messages": messages,
                        "temperature": 0.7,
                        "max_tokens": 512,
                        "stream": True,
                    }
                    if openai_tools:
                        payload["tools"] = openai_tools
                        payload["tool_choice"] = "auto"

                    # Accumulate tool calls from streaming deltas
                    accumulated_tool_calls: dict[int, dict] = {}

                    async with client.stream(
                        "POST", _GROQ_CHAT_URL,
                        headers=self._headers(), json=payload,
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
                            except (json.JSONDecodeError, KeyError, IndexError):
                                continue

                            # Accumulate tool calls from delta
                            if "tool_calls" in delta:
                                for tc_delta in delta["tool_calls"]:
                                    idx = tc_delta["index"]
                                    fn_d = tc_delta.get("function", {})
                                    if idx not in accumulated_tool_calls:
                                        accumulated_tool_calls[idx] = {
                                            "id": tc_delta.get("id", ""),
                                            "function": {
                                                "name": fn_d.get("name", ""),
                                                "arguments": "",
                                            },
                                        }
                                    else:
                                        if tc_delta.get("id"):
                                            accumulated_tool_calls[idx]["id"] = tc_delta["id"]
                                        if fn_d.get("name"):
                                            acc_fn = accumulated_tool_calls[idx]["function"]
                                            acc_fn["name"] = fn_d["name"]
                                    # Append argument chunks
                                    if fn_d.get("arguments"):
                                        acc_fn = accumulated_tool_calls[idx]["function"]
                                        acc_fn["arguments"] += fn_d["arguments"]

                            # Yield text content
                            content = delta.get("content", "")
                            if content:
                                yield content

                    # After stream ends: check if we got tool calls
                    if accumulated_tool_calls:
                        tool_call_count += 1
                        tool_calls_list = [
                            accumulated_tool_calls[i]
                            for i in sorted(accumulated_tool_calls)
                        ]
                        logger.info(
                            "Groq stream tool call round #%d: %d tool(s)",
                            tool_call_count, len(tool_calls_list),
                        )

                        # Add assistant message with tool_calls
                        messages.append({
                            "role": "assistant",
                            "content": "",
                            "tool_calls": tool_calls_list,
                        })

                        # Execute tools and add results
                        tool_results = await _execute_tool_calls(tool_calls_list)
                        messages.extend(tool_results)
                        # Continue loop → new stream with tool results
                        continue

                    # No tool calls — streaming done
                    break

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
