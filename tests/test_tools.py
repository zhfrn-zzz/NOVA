"""Tests for NOVA tools: time_date, system_control, memory, and registry."""

import json
from unittest.mock import patch

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
        expected = [
            "get_current_time", "get_current_date", "get_current_datetime",
            "volume_up", "volume_down", "mute_unmute",
            "play_pause_media", "next_track", "previous_track",
            "open_app", "open_browser", "open_url",
            "open_terminal", "open_file_manager",
            "lock_screen", "shutdown_pc", "restart_pc", "sleep_pc",
            "take_screenshot", "set_timer",
            "web_search", "remember_fact", "recall_facts",
        ]
        for name in expected:
            assert name in names, f"{name!r} not in registry"

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


class TestWebSearchTool:
    @pytest.mark.asyncio
    async def test_web_search_returns_string(self):
        from nova.tools.web_search import web_search

        result = await web_search("Python programming language")
        assert isinstance(result, str)
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_web_search_via_registry(self):
        result = await execute_tool("web_search", {"query": "Python programming"})
        assert isinstance(result, str)
        assert len(result) > 0

    def test_web_search_declared_in_registry(self):
        names = get_all_tool_names()
        assert "web_search" in names

    def test_web_search_declaration_has_query_param(self):
        tools = get_tool_declarations()
        for tool in tools:
            for fn_decl in tool.function_declarations:
                if fn_decl.name == "web_search":
                    schema = fn_decl.parameters_json_schema
                    assert "query" in schema["properties"]
                    assert "query" in schema["required"]
                    return
        pytest.fail("web_search declaration not found")


class TestUserMemory:
    """Tests for persistent user memory (UserMemory class)."""

    @pytest.fixture(autouse=True)
    def _use_tmp_memory(self, tmp_path):
        """Redirect memory file to a temp dir for test isolation."""
        mem_file = tmp_path / "memory.json"
        with (
            patch("nova.memory.persistent._MEMORY_FILE", mem_file),
            patch("nova.memory.persistent._MEMORY_DIR", tmp_path),
            patch("nova.memory.persistent._instance", None),
        ):
            yield mem_file

    def test_add_and_get_facts(self):
        from nova.memory.persistent import UserMemory

        mem = UserMemory()
        mem.add_fact("name", "Zhafran")
        mem.add_fact("location", "Bekasi")
        facts = mem.get_facts()
        assert facts == {"name": "Zhafran", "location": "Bekasi"}

    def test_get_fact_single(self):
        from nova.memory.persistent import UserMemory

        mem = UserMemory()
        mem.add_fact("hobby", "guitar")
        assert mem.get_fact("hobby") == "guitar"
        assert mem.get_fact("nonexistent") is None

    def test_remove_fact(self):
        from nova.memory.persistent import UserMemory

        mem = UserMemory()
        mem.add_fact("color", "blue")
        assert mem.remove_fact("color") is True
        assert mem.remove_fact("color") is False
        assert mem.get_facts() == {}

    def test_clear(self):
        from nova.memory.persistent import UserMemory

        mem = UserMemory()
        mem.add_fact("a", "1")
        mem.add_fact("b", "2")
        mem.clear()
        assert mem.get_facts() == {}
        assert mem.fact_count == 0

    def test_persistence_across_instances(self, _use_tmp_memory):
        from nova.memory.persistent import UserMemory

        mem1 = UserMemory()
        mem1.add_fact("name", "Zhafran")

        # New instance reads from same file
        mem2 = UserMemory()
        assert mem2.get_fact("name") == "Zhafran"

    def test_key_normalized_to_lowercase(self):
        from nova.memory.persistent import UserMemory

        mem = UserMemory()
        mem.add_fact("Name", "Zhafran")
        assert mem.get_fact("name") == "Zhafran"

    def test_file_written_as_json(self, _use_tmp_memory):
        from nova.memory.persistent import UserMemory

        mem = UserMemory()
        mem.add_fact("city", "Jakarta")
        data = json.loads(_use_tmp_memory.read_text(encoding="utf-8"))
        assert data == {"city": "Jakarta"}

    @pytest.mark.asyncio
    async def test_remember_fact_tool(self):
        from nova.memory.persistent import get_user_memory, remember_fact

        result = await remember_fact("name", "Zhafran")
        assert "Zhafran" in result
        assert get_user_memory().get_fact("name") == "Zhafran"

    @pytest.mark.asyncio
    async def test_recall_facts_tool(self):
        from nova.memory.persistent import get_user_memory, recall_facts

        get_user_memory().add_fact("name", "Zhafran")
        result = await recall_facts()
        assert "name=Zhafran" in result

    @pytest.mark.asyncio
    async def test_recall_facts_empty(self):
        from nova.memory.persistent import recall_facts

        result = await recall_facts()
        assert "Belum ada" in result

    def test_memory_tools_in_registry(self):
        names = get_all_tool_names()
        assert "remember_fact" in names
        assert "recall_facts" in names

    @pytest.mark.asyncio
    async def test_remember_fact_via_registry(self):
        result = await execute_tool("remember_fact", {"key": "test", "value": "123"})
        assert isinstance(result, str)
        assert "123" in result


class TestWakeWordBeepGeneration:
    def test_generate_beep_returns_wav_bytes(self):
        from nova.audio.wake_word import generate_beep
        beep = generate_beep()
        assert isinstance(beep, bytes)
        # WAV files start with "RIFF"
        assert beep[:4] == b"RIFF"
        # Should be more than just a header
        assert len(beep) > 44
