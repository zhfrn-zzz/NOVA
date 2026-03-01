"""Tests for heartbeat orchestrator integration — notification context, delivery, and lifecycle."""

import sys
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from nova.heartbeat.queue import Notification, NotificationQueue, Urgency
from nova.memory.prompt_assembler import PromptAssembler

# ── Prompt Assembler: Notification Context ─────────────────────────────


class TestNotificationContext:
    """Test notification context injection in PromptAssembler."""

    @pytest.fixture(autouse=True)
    def _setup(self, tmp_path):
        self.assembler = PromptAssembler(prompts_dir=tmp_path)

    def test_notification_context_in_build(self):
        """Notification context should appear in the assembled prompt."""
        self.assembler.set_notification_context("Remind the user: Ujian jam 8")
        prompt = self.assembler.build()
        assert "Pending notifications to deliver" in prompt
        assert "Remind the user: Ujian jam 8" in prompt

    def test_notification_context_consumed_after_build(self):
        """Notification context should be cleared after build()."""
        self.assembler.set_notification_context("Test notification")
        self.assembler.build()
        # Second build should have no notification context
        prompt2 = self.assembler.build()
        assert "Pending notifications" not in prompt2

    def test_notification_and_memory_context_coexist(self):
        """Both memory and notification context should appear together."""
        self.assembler.set_memory_context("User likes coffee")
        self.assembler.set_notification_context("Morning greeting")
        prompt = self.assembler.build()
        assert "Relevant memories" in prompt
        assert "Pending notifications" in prompt

    def test_no_notification_section_when_empty(self):
        """No notification section when no context is set."""
        prompt = self.assembler.build()
        assert "Pending notifications" not in prompt


# ── Orchestrator: format_notifications ──────────────────────────────
# We need to mock psutil before importing orchestrator since
# nova.tools.system_info depends on it.


@pytest.fixture(scope="module")
def orchestrator_module():
    """Import orchestrator module with psutil mocked out."""
    psutil_mock = MagicMock()
    with patch.dict(sys.modules, {"psutil": psutil_mock}):
        from nova import orchestrator as orch_mod
    return orch_mod


class TestFormatNotifications:
    """Test Orchestrator._format_notifications static method."""

    def test_morning_greeting_token(self, orchestrator_module):
        """__morning_greeting__ should produce greeting instruction."""
        orch_cls = orchestrator_module.Orchestrator

        notif = Notification(
            message="__morning_greeting__",
            urgency=Urgency.PASSIVE,
            source="rule",
            created_at=datetime.now(),
        )
        result = orch_cls._format_notifications([notif])
        assert "morning greeting" in result.lower()

    def test_sleep_reminder_token(self, orchestrator_module):
        """__sleep_reminder__ should produce sleep reminder instruction."""
        orch_cls = orchestrator_module.Orchestrator

        notif = Notification(
            message="__sleep_reminder__",
            urgency=Urgency.GENTLE,
            source="rule",
            created_at=datetime.now(),
        )
        result = orch_cls._format_notifications([notif])
        assert "late" in result.lower() or "rest" in result.lower()

    def test_regular_message(self, orchestrator_module):
        """Regular message should be formatted as reminder."""
        orch_cls = orchestrator_module.Orchestrator

        notif = Notification(
            message="Ujian besok jam 8",
            urgency=Urgency.GENTLE,
            source="reminder",
            created_at=datetime.now(),
        )
        result = orch_cls._format_notifications([notif])
        assert "Ujian besok jam 8" in result

    def test_multiple_notifications(self, orchestrator_module):
        """Multiple notifications should be joined with newlines."""
        orch_cls = orchestrator_module.Orchestrator

        notifs = [
            Notification(
                message="__morning_greeting__",
                urgency=Urgency.PASSIVE,
                source="rule",
                created_at=datetime.now(),
            ),
            Notification(
                message="Meeting jam 3",
                urgency=Urgency.GENTLE,
                source="reminder",
                created_at=datetime.now(),
            ),
        ]
        result = orch_cls._format_notifications(notifs)
        lines = result.strip().split("\n")
        assert len(lines) == 2


# ── Orchestrator: Heartbeat Lifecycle ───────────────────────────────


class TestHeartbeatLifecycle:
    """Test heartbeat scheduler start/stop via orchestrator."""

    def test_scheduler_starts_when_enabled(self, orchestrator_module):
        """HeartbeatScheduler.start() should be called when heartbeat_enabled=True."""
        with patch.object(orchestrator_module, "get_config") as mock_config, \
             patch.object(orchestrator_module, "get_memory_store"), \
             patch.object(orchestrator_module, "get_embedder", return_value=None), \
             patch.object(orchestrator_module, "get_prompt_assembler"), \
             patch.object(orchestrator_module, "GeminiProvider"), \
             patch.object(orchestrator_module, "GroqWhisperProvider"), \
             patch.object(orchestrator_module, "EdgeTTSProvider"), \
             patch.object(orchestrator_module, "ProviderRouter"), \
             patch.object(orchestrator_module, "ConversationManager"), \
             patch.object(orchestrator_module, "StreamingTTSPlayer"), \
             patch.object(orchestrator_module, "get_tool_declarations"), \
             patch.object(orchestrator_module, "HeartbeatScheduler") as mock_scheduler:
            cfg = MagicMock()
            cfg.heartbeat_enabled = True
            cfg.gemini_api_key = "test-key"
            cfg.groq_api_key = ""
            cfg.cloudflare_account_id = ""
            cfg.cloudflare_api_token = ""
            cfg.google_cloud_tts_key_path = ""
            cfg.embedding_enabled = False
            mock_config.return_value = cfg

            orchestrator_module.Orchestrator()
            mock_scheduler.return_value.start.assert_called_once()

    def test_scheduler_stop_on_orchestrator_stop(self, orchestrator_module):
        """orchestrator.stop() should call scheduler.stop()."""
        with patch.object(orchestrator_module, "get_config") as mock_config, \
             patch.object(orchestrator_module, "get_memory_store"), \
             patch.object(orchestrator_module, "get_embedder", return_value=None), \
             patch.object(orchestrator_module, "get_prompt_assembler"), \
             patch.object(orchestrator_module, "GeminiProvider"), \
             patch.object(orchestrator_module, "GroqWhisperProvider"), \
             patch.object(orchestrator_module, "EdgeTTSProvider"), \
             patch.object(orchestrator_module, "ProviderRouter"), \
             patch.object(orchestrator_module, "ConversationManager"), \
             patch.object(orchestrator_module, "StreamingTTSPlayer"), \
             patch.object(orchestrator_module, "get_tool_declarations"), \
             patch.object(orchestrator_module, "HeartbeatScheduler") as mock_scheduler:
            cfg = MagicMock()
            cfg.heartbeat_enabled = True
            cfg.gemini_api_key = "test-key"
            cfg.groq_api_key = ""
            cfg.cloudflare_account_id = ""
            cfg.cloudflare_api_token = ""
            cfg.google_cloud_tts_key_path = ""
            cfg.embedding_enabled = False
            mock_config.return_value = cfg

            orch = orchestrator_module.Orchestrator()
            orch.stop()
            mock_scheduler.return_value.stop.assert_called_once()


# ── Orchestrator: Passive Injection ────────────────────────────────


class TestPassiveInjection:
    """Test that passive notifications get injected into LLM context."""

    def test_inject_passive_calls_assembler(self, orchestrator_module):
        """_inject_passive_notifications should call set_notification_context."""
        orch_cls = orchestrator_module.Orchestrator

        # Create a minimal orchestrator mock that has the real method
        orch = MagicMock(spec=orch_cls)
        orch._notification_queue = NotificationQueue()
        orch._inject_passive_notifications = (
            orch_cls._inject_passive_notifications.__get__(orch)
        )
        orch._format_notifications = orch_cls._format_notifications

        # Push a passive notification
        orch._notification_queue.push(Notification(
            message="__morning_greeting__",
            urgency=Urgency.PASSIVE,
            source="rule",
            created_at=datetime.now(),
        ))

        with patch.object(orchestrator_module, "get_prompt_assembler") as mock_assembler:
            orch._inject_passive_notifications()
            mock_assembler.return_value.set_notification_context.assert_called_once()
            call_arg = mock_assembler.return_value.set_notification_context.call_args[0][0]
            assert "morning greeting" in call_arg.lower()

    def test_no_injection_when_queue_empty(self, orchestrator_module):
        """No call to set_notification_context when queue is empty."""
        orch_cls = orchestrator_module.Orchestrator

        orch = MagicMock(spec=orch_cls)
        orch._notification_queue = NotificationQueue()
        orch._inject_passive_notifications = (
            orch_cls._inject_passive_notifications.__get__(orch)
        )

        with patch.object(orchestrator_module, "get_prompt_assembler") as mock_assembler:
            orch._inject_passive_notifications()
            mock_assembler.return_value.set_notification_context.assert_not_called()


# ── Orchestrator: Notification Queue Property ──────────────────────


class TestNotificationQueueProperty:
    """Test that notification_queue property works correctly."""

    def test_queue_accessible(self, orchestrator_module):
        """notification_queue property should return the internal queue."""
        with patch.object(orchestrator_module, "get_config") as mock_config, \
             patch.object(orchestrator_module, "get_memory_store"), \
             patch.object(orchestrator_module, "get_embedder", return_value=None), \
             patch.object(orchestrator_module, "get_prompt_assembler"), \
             patch.object(orchestrator_module, "GeminiProvider"), \
             patch.object(orchestrator_module, "GroqWhisperProvider"), \
             patch.object(orchestrator_module, "EdgeTTSProvider"), \
             patch.object(orchestrator_module, "ProviderRouter"), \
             patch.object(orchestrator_module, "ConversationManager"), \
             patch.object(orchestrator_module, "StreamingTTSPlayer"), \
             patch.object(orchestrator_module, "get_tool_declarations"), \
             patch.object(orchestrator_module, "HeartbeatScheduler"):
            cfg = MagicMock()
            cfg.heartbeat_enabled = False
            cfg.gemini_api_key = "test"
            cfg.groq_api_key = ""
            cfg.cloudflare_account_id = ""
            cfg.cloudflare_api_token = ""
            cfg.google_cloud_tts_key_path = ""
            cfg.embedding_enabled = False
            mock_config.return_value = cfg

            orch = orchestrator_module.Orchestrator()
            assert isinstance(orch.notification_queue, NotificationQueue)
