"""Tests for PromptAssembler — file-based prompt system."""

import pytest

from nova.memory.prompt_assembler import PromptAssembler


@pytest.fixture
def prompts_dir(tmp_path):
    """Create a temporary prompts directory."""
    d = tmp_path / "prompts"
    d.mkdir()
    return d


class TestPromptAssembler:
    def test_creates_default_files(self, tmp_path):
        d = tmp_path / "prompts"
        PromptAssembler(prompts_dir=d)
        assert (d / "SOUL.md").exists()
        assert (d / "RULES.md").exists()
        assert (d / "USER.md").exists()

    def test_does_not_overwrite_existing_files(self, prompts_dir):
        (prompts_dir / "SOUL.md").write_text("Custom soul", encoding="utf-8")
        PromptAssembler(prompts_dir=prompts_dir)
        content = (prompts_dir / "SOUL.md").read_text(encoding="utf-8")
        assert content == "Custom soul"

    def test_build_includes_soul(self, prompts_dir):
        (prompts_dir / "SOUL.md").write_text("I am NOVA.", encoding="utf-8")
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        prompt = assembler.build(datetime_str="test-time")
        assert "I am NOVA." in prompt

    def test_build_includes_rules(self, prompts_dir):
        (prompts_dir / "RULES.md").write_text("Be concise.", encoding="utf-8")
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        prompt = assembler.build(datetime_str="test-time")
        assert "Be concise." in prompt

    def test_build_includes_user_with_header(self, prompts_dir):
        (prompts_dir / "USER.md").write_text("Name: Test", encoding="utf-8")
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        prompt = assembler.build(datetime_str="test-time")
        assert "About the user:" in prompt
        assert "Name: Test" in prompt

    def test_build_includes_datetime(self, prompts_dir):
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        prompt = assembler.build(datetime_str="Senin, 26 Februari 2026")
        assert "Current date and time: Senin, 26 Februari 2026" in prompt

    def test_build_auto_generates_datetime(self, prompts_dir):
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        prompt = assembler.build()
        assert "Current date and time: Sekarang:" in prompt

    def test_build_includes_memory_context(self, prompts_dir):
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        prompt = assembler.build(
            memory_context="hobby=guitar, pet=cat",
            datetime_str="test-time",
        )
        assert "Relevant memories:" in prompt
        assert "hobby=guitar" in prompt

    def test_build_omits_empty_memory(self, prompts_dir):
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        prompt = assembler.build(datetime_str="test-time")
        assert "Relevant memories:" not in prompt

    def test_caching_reads_file_once(self, prompts_dir):
        (prompts_dir / "SOUL.md").write_text("V1", encoding="utf-8")
        assembler = PromptAssembler(prompts_dir=prompts_dir)

        # First read
        p1 = assembler.build(datetime_str="t")
        assert "V1" in p1

        # Modify file without changing mtime (same content size)
        # mtime should be the same within the same test run
        p2 = assembler.build(datetime_str="t")
        assert "V1" in p2

    def test_hot_reload_on_mtime_change(self, prompts_dir):
        soul = prompts_dir / "SOUL.md"
        soul.write_text("V1", encoding="utf-8")
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        p1 = assembler.build(datetime_str="t")
        assert "V1" in p1

        # Force mtime change
        import time
        time.sleep(0.05)
        soul.write_text("V2", encoding="utf-8")

        p2 = assembler.build(datetime_str="t")
        assert "V2" in p2

    def test_missing_file_returns_empty(self, prompts_dir):
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        # Delete SOUL.md
        (prompts_dir / "SOUL.md").unlink()
        # Should still build without error
        prompt = assembler.build(datetime_str="t")
        assert prompt  # contains at least rules+user+datetime

    def test_update_user_profile(self, prompts_dir):
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        assembler.update_user_profile("Favorite color: blue")
        content = (prompts_dir / "USER.md").read_text(encoding="utf-8")
        assert "Favorite color: blue" in content

    def test_update_user_profile_appends(self, prompts_dir):
        (prompts_dir / "USER.md").write_text("Name: Test", encoding="utf-8")
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        assembler.update_user_profile("Age: 17")
        content = (prompts_dir / "USER.md").read_text(encoding="utf-8")
        assert "Name: Test" in content
        assert "Age: 17" in content

    def test_prompts_dir_property(self, prompts_dir):
        assembler = PromptAssembler(prompts_dir=prompts_dir)
        assert assembler.prompts_dir == prompts_dir
