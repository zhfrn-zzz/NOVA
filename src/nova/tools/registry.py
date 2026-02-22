"""Tool registry — central catalogue of all NOVA tools for LLM function calling.

Defines FunctionDeclaration schemas for Gemini and dispatches tool calls
to the correct implementation.
"""

import logging

from google.genai import types

from nova.tools import system_control, time_date

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Function declarations for the Gemini function-calling API
# ---------------------------------------------------------------------------

_FUNCTION_DECLARATIONS = [
    types.FunctionDeclaration(
        name="get_current_time",
        description=(
            "Get the current local time. Use this when the user asks what time it is, "
            "jam berapa, or any time-related question."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_current_date",
        description=(
            "Get the current local date. Use this when the user asks what today's date is, "
            "tanggal berapa, or any date-related question."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_current_datetime",
        description=(
            "Get both the current local date and time together."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="volume_up",
        description=(
            "Increase the system volume. Use when the user says 'volume up', "
            "'naikkan volume', 'louder', 'kerasin', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="volume_down",
        description=(
            "Decrease the system volume. Use when the user says 'volume down', "
            "'kecilkan volume', 'turunkan volume', 'quieter', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="open_browser",
        description=(
            "Open the default web browser. Use when the user says 'open browser', "
            "'buka browser', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="open_terminal",
        description=(
            "Open a terminal window. Use when the user says 'open terminal', "
            "'buka terminal', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="lock_screen",
        description=(
            "Lock the computer screen. Use when the user says 'lock screen', "
            "'kunci layar', 'lock komputer', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
]

# Map function names → async callables
_TOOL_IMPLEMENTATIONS: dict[str, object] = {
    "get_current_time": time_date.get_current_time,
    "get_current_date": time_date.get_current_date,
    "get_current_datetime": time_date.get_current_datetime,
    "volume_up": system_control.volume_up,
    "volume_down": system_control.volume_down,
    "open_browser": system_control.open_browser,
    "open_terminal": system_control.open_terminal,
    "lock_screen": system_control.lock_screen,
}


def get_tool_declarations() -> list[types.Tool]:
    """Return the list of Tool objects for Gemini function calling.

    Returns:
        A list containing a single Tool with all function declarations.
    """
    return [types.Tool(function_declarations=_FUNCTION_DECLARATIONS)]


async def execute_tool(name: str, args: dict | None = None) -> str:
    """Execute a tool by name and return its result.

    Args:
        name: The function name as returned by the LLM function call.
        args: Arguments dict (most tools take none).

    Returns:
        The tool's result as a string.

    Raises:
        ValueError: If the tool name is unknown.
    """
    impl = _TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        raise ValueError(f"Unknown tool: {name!r}")

    logger.info("Executing tool: %s(%s)", name, args or "")
    result = await impl(**(args or {}))
    logger.info("Tool %s result: %s", name, result)
    return result


def get_all_tool_names() -> list[str]:
    """Return all registered tool function names.

    Returns:
        Sorted list of tool name strings.
    """
    return sorted(_TOOL_IMPLEMENTATIONS.keys())
