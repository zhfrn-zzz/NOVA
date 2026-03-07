"""Tests for messaging integration — Telegram bot, WhatsApp client, and text mode."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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

    def _make_client(self, allowed_jids: list[str] | None = None):
        """Create a NovaWhatsAppClient with mocked orchestrator."""
        from nova.messaging.whatsapp_client import NovaWhatsAppClient

        orchestrator = AsyncMock()
        orchestrator.handle_interaction = AsyncMock(return_value="test response")
        client = NovaWhatsAppClient(
            orchestrator=orchestrator,
            allowed_jids=allowed_jids,
            port=0,  # Won't actually bind
        )
        return client, orchestrator

    @pytest.mark.asyncio
    async def test_incoming_authorized_bare_number(self):
        """Bare phone number in allowed list matches @s.whatsapp.net sender."""
        client, orchestrator = self._make_client(["628123456789"])

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "628123456789@s.whatsapp.net",
            "text": "nyalakan AC",
        })

        resp = await client._handle_incoming(request)
        data = resp.body
        assert b"test response" in data
        orchestrator.handle_interaction.assert_awaited_once_with(
            "nyalakan AC", mode="text",
        )

    @pytest.mark.asyncio
    async def test_incoming_authorized_lid(self):
        """@lid JID in allowed list matches @lid sender."""
        client, orchestrator = self._make_client(["134711984783457@lid"])

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "134711984783457@lid",
            "text": "hello",
        })

        resp = await client._handle_incoming(request)
        assert b"test response" in resp.body

    @pytest.mark.asyncio
    async def test_incoming_authorized_mixed_formats(self):
        """Both @lid and @s.whatsapp.net in allowed list works."""
        client, orchestrator = self._make_client([
            "134711984783457@lid",
            "628123456789@s.whatsapp.net",
        ])

        # Message from @lid
        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "134711984783457@lid",
            "text": "hello lid",
        })
        await client._handle_incoming(request)
        assert orchestrator.handle_interaction.await_count == 1

        # Message from @s.whatsapp.net
        request2 = MagicMock()
        request2.json = AsyncMock(return_value={
            "sender": "628123456789@s.whatsapp.net",
            "text": "hello snet",
        })
        await client._handle_incoming(request2)
        assert orchestrator.handle_interaction.await_count == 2

    @pytest.mark.asyncio
    async def test_incoming_unauthorized(self):
        """Unauthorized sender gets empty reply."""
        client, orchestrator = self._make_client(["628123456789@s.whatsapp.net"])

        request = MagicMock()
        request.json = AsyncMock(return_value={
            "sender": "628999999999@s.whatsapp.net",
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
