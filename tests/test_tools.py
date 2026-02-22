"""Tests for NOVA tools: time_date, system_control, and registry."""

import pytest

from nova.tools import time_date
from nova.tools.registry import execute_tool, get_all_tool_names, get_tool_declarations


class TestTimeDateTools:
    @pytest.mark.asyncio
    async def test_get_current_time_format(self):
        result = await time_date.get_current_time()
        # Should be in HH:MM format
        assert ":" in result
        parts = result.split(":")
        assert len(parts) == 2
        hour, minute = int(parts[0]), int(parts[1])
        assert 0 <= hour <= 23
        assert 0 <= minute <= 59

    @pytest.mark.asyncio
    async def test_get_current_date_format(self):
        result = await time_date.get_current_date()
        # Should contain a day name and year
        assert "," in result
        assert "20" in result  # year 20xx

    @pytest.mark.asyncio
    async def test_get_current_datetime_format(self):
        result = await time_date.get_current_datetime()
        # Should contain both date and time
        assert "pukul" in result
        assert ":" in result


class TestToolRegistry:
    def test_get_tool_declarations_returns_tools(self):
        tools = get_tool_declarations()
        assert len(tools) >= 1
        # Each tool should be a google.genai types.Tool
        tool = tools[0]
        assert hasattr(tool, "function_declarations")
        assert len(tool.function_declarations) > 0

    def test_get_all_tool_names(self):
        names = get_all_tool_names()
        assert "get_current_time" in names
        assert "get_current_date" in names
        assert "get_current_datetime" in names
        assert "volume_up" in names
        assert "volume_down" in names
        assert "open_browser" in names
        assert "open_terminal" in names
        assert "lock_screen" in names

    @pytest.mark.asyncio
    async def test_execute_tool_time(self):
        result = await execute_tool("get_current_time")
        assert isinstance(result, str)
        assert ":" in result

    @pytest.mark.asyncio
    async def test_execute_tool_date(self):
        result = await execute_tool("get_current_date")
        assert isinstance(result, str)
        assert len(result) > 5

    @pytest.mark.asyncio
    async def test_execute_tool_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown tool"):
            await execute_tool("nonexistent_tool")

    @pytest.mark.asyncio
    async def test_execute_tool_with_empty_args(self):
        result = await execute_tool("get_current_time", {})
        assert isinstance(result, str)

    def test_all_declared_tools_have_implementations(self):
        """Every declared function should have a matching implementation."""
        tools = get_tool_declarations()
        names = get_all_tool_names()
        for tool in tools:
            for fn_decl in tool.function_declarations:
                assert fn_decl.name in names, (
                    f"Declared function {fn_decl.name!r} has no implementation"
                )


class TestWakeWordBeepGeneration:
    def test_generate_beep_returns_wav_bytes(self):
        from nova.audio.wake_word import generate_beep
        beep = generate_beep()
        assert isinstance(beep, bytes)
        # WAV files start with "RIFF"
        assert beep[:4] == b"RIFF"
        # Should be more than just a header
        assert len(beep) > 44
