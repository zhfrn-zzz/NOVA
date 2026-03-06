"""Tests for messaging integration — Telegram bot, WhatsApp client, and text mode."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from nova.messaging.formatter import format_for_telegram, format_for_whatsapp

# ---------------------------------------------------------------------------
# Telegram Bot Tests
# ---------------------------------------------------------------------------


class TestTelegramBotAuth:
    """Test NovaTelegramBot authorization checks."""

    def _make_bot(self, allowed_users: list[int]):
        """Create a NovaTelegramBot instance with mocked dependencies."""
        with patch("nova.messaging.telegram_bot.Bot"), \
             patch("nova.messaging.telegram_bot.Dispatcher"):
            from nova.messaging.telegram_bot import NovaTelegramBot

            orchestrator = AsyncMock()
            return NovaTelegramBot(
                token="fake:token",
                allowed_users=allowed_users,
                orchestrator=orchestrator,
            )

    def _make_message(self, user_id: int, username: str = "testuser"):
        """Create a mock Telegram message."""
        msg = MagicMock()
        msg.from_user = MagicMock()
        msg.from_user.id = user_id
        msg.from_user.username = username
        return msg

    def test_authorized_user_allowed(self):
        bot = self._make_bot([123, 456])
        msg = self._make_message(123)
        assert bot._is_authorized(msg) is True

    def test_unauthorized_user_rejected(self):
        bot = self._make_bot([123, 456])
        msg = self._make_message(999)
        assert bot._is_authorized(msg) is False

    def test_empty_allowed_list_rejects_all(self):
        bot = self._make_bot([])
        msg = self._make_message(123)
        assert bot._is_authorized(msg) is False

    def test_none_user_rejected(self):
        bot = self._make_bot([123])
        msg = MagicMock()
        msg.from_user = None
        assert bot._is_authorized(msg) is False


# ---------------------------------------------------------------------------
# WhatsApp Client Tests
# ---------------------------------------------------------------------------


class TestWhatsAppClient:
    """Test NovaWhatsAppClient authorization and message handling."""

    def _make_client(self, allowed_numbers: list[str] | None = None):
        """Create a NovaWhatsAppClient with mocked orchestrator."""
        from nova.messaging.whatsapp_client import NovaWhatsAppClient

        orchestrator = AsyncMock()
        orchestrator.handle_interaction = AsyncMock(return_value="test response")
        client = NovaWhatsAppClient(
            orchestrator=orchestrator,
            allowed_numbers=allowed_numbers,
            port=0,  # Won't actually bind
        )
        return client, orchestrator

    @pytest.mark.asyncio
    async def test_incoming_authorized(self):
        """Authorized sender gets a response."""
        client, orchestrator = self._make_client(["628123456789"])

        # Simulate request
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "628123456789",
            "text": "nyalakan AC",
            "jid": "628123456789@s.whatsapp.net",
        })

        resp = await client._handle_incoming(request)
        data = resp.body
        assert b"test response" in data
        orchestrator.handle_interaction.assert_awaited_once_with(
            "nyalakan AC", mode="text",
        )

    @pytest.mark.asyncio
    async def test_incoming_unauthorized(self):
        """Unauthorized sender gets empty reply."""
        client, orchestrator = self._make_client(["628123456789"])

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "628999999999",
            "text": "hack the planet",
        })

        resp = await client._handle_incoming(request)
        assert b'"reply": ""' in resp.body or resp.body == b'{"reply": ""}'
        orchestrator.handle_interaction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_incoming_empty_text(self):
        """Empty text gets empty reply."""
        client, orchestrator = self._make_client()

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "628123456789",
            "text": "",
        })

        await client._handle_incoming(request)
        orchestrator.handle_interaction.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_incoming_no_allowed_list_accepts_all(self):
        """No allowed list means all senders are accepted."""
        client, orchestrator = self._make_client(None)

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "628999999999",
            "text": "hello",
        })

        await client._handle_incoming(request)
        orchestrator.handle_interaction.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_health_check_bridge_down(self):
        """Health check returns False when bridge is not running."""
        client, _ = self._make_client()
        result = await client.check_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_send_message_bridge_down(self):
        """send_message returns False when bridge is not running."""
        client, _ = self._make_client()
        result = await client.send_message("628123456789", "hello")
        assert result is False


# ---------------------------------------------------------------------------
# Formatter Tests
# ---------------------------------------------------------------------------


class TestFormatter:
    """Test response formatting functions."""

    def test_format_for_telegram_passthrough(self):
        text = "AC sudah dinyalakan di suhu 24 derajat."
        assert format_for_telegram(text) == text

    def test_format_for_whatsapp_passthrough(self):
        text = "Besok diprediksi hujan dengan kemungkinan 80%."
        assert format_for_whatsapp(text) == text


# ---------------------------------------------------------------------------
# Prompt Assembler Text Mode Tests
# ---------------------------------------------------------------------------


class TestPromptAssemblerTextMode:
    """Test prompt assembler interaction mode adjustment."""

    def test_set_interaction_mode(self):
        from nova.memory.prompt_assembler import PromptAssembler

        assembler = PromptAssembler(prompts_dir="/tmp/nova_test_prompts_msg")
        assembler.set_interaction_mode("text")
        assert assembler._interaction_mode == "text"

    def test_adapt_rules_removes_voice_lines(self):
        from nova.memory.prompt_assembler import PromptAssembler

        rules = (
            "Response rules:\n"
            "- Keep responses between 20-50 words unless user asks for detail.\n"
            "- Responses will be spoken aloud — plain text only.\n"
            "- No markdown, bullet points, asterisks, emoji, exclamation marks.\n"
            "- Default to Indonesian unless user speaks English."
        )

        result = PromptAssembler._adapt_rules_for_text(rules)

        assert "spoken aloud" not in result
        assert "No markdown" not in result
        assert "Keep responses between 20-50 words" in result
        assert "text message" in result
        assert "mobile messaging" in result

    def test_interaction_mode_resets_after_build(self):
        from nova.memory.prompt_assembler import PromptAssembler

        assembler = PromptAssembler(prompts_dir="/tmp/nova_test_prompts_msg2")
        assembler.set_interaction_mode("text")
        assembler.build()
        assert assembler._interaction_mode == "voice"

    def test_voice_mode_preserves_rules(self):
        from nova.memory.prompt_assembler import PromptAssembler

        assembler = PromptAssembler(prompts_dir="/tmp/nova_test_prompts_msg3")
        # Default mode is "voice"
        prompt = assembler.build()
        assert "spoken aloud" in prompt


# ---------------------------------------------------------------------------
# Config Tests
# ---------------------------------------------------------------------------


class TestMessagingConfig:
    """Test messaging config fields."""

    def test_telegram_defaults(self):
        from nova.config import NovaConfig

        config = NovaConfig(
            _env_file=None,
            gemini_api_key="test",
        )
        assert config.telegram_bot_token == ""
        assert config.telegram_allowed_users == []

    def test_whatsapp_defaults(self):
        from nova.config import NovaConfig

        config = NovaConfig(
            _env_file=None,
            gemini_api_key="test",
        )
        assert config.whatsapp_enabled is False
        assert config.whatsapp_allowed_jids == []
