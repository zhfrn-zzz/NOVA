"""Tool registry — central catalogue of all NOVA tools for LLM function calling.

Defines FunctionDeclaration schemas for Gemini and dispatches tool calls
to the correct implementation.
"""

import logging

from google.genai import types

from nova.memory.persistent import recall_facts, remember_fact
from nova.tools import system_control, time_date, web_search

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Function declarations for the Gemini function-calling API
# ---------------------------------------------------------------------------

_FUNCTION_DECLARATIONS = [
    # ── Time / Date ──────────────────────────────────────────────────
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
    # ── Volume ───────────────────────────────────────────────────────
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
        name="mute_unmute",
        description=(
            "Toggle system audio mute/unmute. Use when the user says 'mute', "
            "'unmute', 'bisukan', 'matikan suara', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Media Controls ───────────────────────────────────────────────
    types.FunctionDeclaration(
        name="play_pause_media",
        description=(
            "Toggle media play/pause. Use when the user says 'play', 'pause', "
            "'putar musik', 'pause musik', 'stop musik', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="next_track",
        description=(
            "Skip to next media track. Use when the user says 'next', 'skip', "
            "'lagu selanjutnya', 'next track', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="previous_track",
        description=(
            "Go to previous media track. Use when the user says 'previous', 'back', "
            "'lagu sebelumnya', 'previous track', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Applications ─────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="open_app",
        description=(
            "Open an application by name. Supports: notepad, calculator, spotify, "
            "discord, whatsapp, vscode, explorer, paint, settings, task manager. "
            "Use when the user says 'buka spotify', 'open notepad', 'buka kalkulator', "
            "etc. For unlisted apps, try the name directly."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "App name, e.g. 'spotify', 'notepad', 'vscode'.",
                },
            },
            "required": ["app_name"],
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
        name="open_url",
        description=(
            "Open a specific URL in the default browser. Use when the user asks "
            "to open a website like 'buka youtube', 'open github.com', etc."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to open, e.g. 'https://youtube.com'.",
                },
            },
            "required": ["url"],
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
        name="open_file_manager",
        description=(
            "Open the file manager (Explorer). Use when the user says 'open explorer', "
            "'buka file manager', 'buka explorer', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── System Power ─────────────────────────────────────────────────
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
    types.FunctionDeclaration(
        name="shutdown_pc",
        description=(
            "Schedule a system shutdown with a countdown. Default 60 seconds delay. "
            "Use when the user says 'shutdown', 'matikan komputer', 'turn off pc'. "
            "Always confirm with the user before executing."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "integer",
                    "description": "Seconds before shutdown (default 60).",
                },
            },
        },
    ),
    types.FunctionDeclaration(
        name="restart_pc",
        description=(
            "Schedule a system restart with a countdown. Default 60 seconds delay. "
            "Use when the user says 'restart', 'restart komputer', 'reboot'. "
            "Always confirm with the user before executing."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "integer",
                    "description": "Seconds before restart (default 60).",
                },
            },
        },
    ),
    types.FunctionDeclaration(
        name="sleep_pc",
        description=(
            "Put the PC to sleep mode. Use when the user says 'sleep', "
            "'tidurkan komputer', 'sleep mode', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Screenshot ───────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="take_screenshot",
        description=(
            "Take a screenshot and save it. Use when the user says 'screenshot', "
            "'ambil screenshot', 'tangkap layar', 'take screenshot', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Timer ────────────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="set_timer",
        description=(
            "Set a countdown timer that shows a notification when done. "
            "Use when the user says 'set timer 5 menit', 'timer 30 detik', "
            "'remind me in 10 minutes', or similar. Convert to seconds."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Timer duration in seconds.",
                },
                "label": {
                    "type": "string",
                    "description": "Description for the notification (default 'Timer').",
                },
            },
            "required": ["seconds"],
        },
    ),
    # ── Web Search ───────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the web for current information. Use this when the user asks about "
            "current events, news, weather, recent facts, or anything you don't know or "
            "that may have changed after your training cutoff. "
            "Examples: 'siapa presiden Indonesia sekarang', 'cuaca Jakarta hari ini', "
            "'berita terbaru', 'what happened today', 'latest news'. "
            "Do NOT use this for time/date queries (use get_current_time instead) or "
            "for general knowledge you already know."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                },
            },
            "required": ["query"],
        },
    ),
    # ── User Memory ──────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="remember_fact",
        description=(
            "Store a fact about the user in persistent memory so you can remember it "
            "across conversations. Use when the user tells you personal information like "
            "their name, location, preferences, hobbies, etc. "
            "Examples: 'nama saya Zhafran', 'I live in Bekasi', 'saya suka kopi', "
            "'ingat bahwa saya alergi kacang', 'my favorite color is blue'. "
            "Choose a short descriptive key and store the value."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "Short identifier for the fact, e.g. 'name', 'location', "
                        "'hobby', 'favorite_food', 'allergy'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The fact value, e.g. 'Zhafran', 'Bekasi', 'guitar'.",
                },
            },
            "required": ["key", "value"],
        },
    ),
    types.FunctionDeclaration(
        name="recall_facts",
        description=(
            "Retrieve all stored facts about the user from persistent memory. "
            "Use when the user asks if you remember something about them, asks 'siapa "
            "nama saya', 'kamu ingat saya?', 'what do you know about me', or similar."
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
    "mute_unmute": system_control.mute_unmute,
    "play_pause_media": system_control.play_pause_media,
    "next_track": system_control.next_track,
    "previous_track": system_control.previous_track,
    "open_app": system_control.open_app,
    "open_browser": system_control.open_browser,
    "open_url": system_control.open_url,
    "open_terminal": system_control.open_terminal,
    "open_file_manager": system_control.open_file_manager,
    "lock_screen": system_control.lock_screen,
    "shutdown_pc": system_control.shutdown_pc,
    "restart_pc": system_control.restart_pc,
    "sleep_pc": system_control.sleep_pc,
    "take_screenshot": system_control.take_screenshot,
    "set_timer": system_control.set_timer,
    "web_search": web_search.web_search,
    "remember_fact": remember_fact,
    "recall_facts": recall_facts,
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
