"""Tests for HeartbeatScheduler — reminder scanning, rules, and urgency logic."""

import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from nova.heartbeat.queue import NotificationQueue, Urgency
from nova.heartbeat.scheduler import HeartbeatScheduler


@pytest.fixture
def mock_config():
    """Create a mock NovaConfig with heartbeat defaults."""
    config = MagicMock()
    config.heartbeat_enabled = True
    config.heartbeat_interval = 60
    config.quiet_hours_start = 23
    config.quiet_hours_end = 6
    config.morning_greeting_enabled = True
    config.sleep_reminder_enabled = True
    config.ambient_presence_threshold = 0.005
    config.chime_volume = 0.3
    config.alert_volume = 0.5
    config.gentle_listen_timeout = 5
    config.gentle_max_retries = 3
    config.gentle_retry_delay = 300
    return config


@pytest.fixture
def mock_store():
    """Create a mock MemoryStore with reminder methods."""
    store = MagicMock()
    store.get_pending_reminders.return_value = []
    store.mark_reminder_delivered.return_value = None
    store.schedule_next_recurrence.return_value = None
    return store


@pytest.fixture
def queue():
    """Create a fresh NotificationQueue."""
    return NotificationQueue()


@pytest.fixture
def scheduler(mock_store, queue, mock_config):
    """Create a HeartbeatScheduler with mocked dependencies."""
    return HeartbeatScheduler(
        memory_store=mock_store,
        notification_queue=queue,
        config=mock_config,
    )


# ── Reminder Checking ────────────────────────────────────────────────


class TestCheckReminders:
    def test_pending_reminder_pushed_to_queue(self, scheduler, mock_store, queue):
        """Pending reminder should be pushed to queue with correct urgency."""
        now = datetime(2026, 3, 1, 14, 0)  # 2pm, not quiet hours
        mock_store.get_pending_reminders.return_value = [{
            "id": 1,
            "message": "Ujian",
            "remind_at": now + timedelta(minutes=3),
            "lead_time": 5,
            "is_alarm": False,
            "urgency": 2,
            "recurring": None,
        }]

        scheduler._check_reminders(now)

        assert queue.size() == 1
        notif = queue.get_next_urgent()
        assert notif.message == "Ujian"
        assert notif.urgency == Urgency.GENTLE
        assert notif.source == "reminder"
        assert notif.reminder_id == 1
        mock_store.mark_reminder_delivered.assert_called_once_with(1)

    def test_quiet_hours_downgrade_non_alarm(self, scheduler, mock_store, queue):
        """During quiet hours, non-alarm reminders → PASSIVE."""
        now = datetime(2026, 3, 1, 23, 30)  # 11:30pm = quiet hours
        mock_store.get_pending_reminders.return_value = [{
            "id": 2,
            "message": "Late reminder",
            "remind_at": now + timedelta(minutes=1),
            "lead_time": 5,
            "is_alarm": False,
            "urgency": 3,  # ACTIVE
            "recurring": None,
        }]

        scheduler._check_reminders(now)

        assert queue.size() == 1
        # Should have been downgraded to PASSIVE
        passive = queue.get_passive()
        assert len(passive) == 1
        assert passive[0].urgency == Urgency.PASSIVE

    def test_alarm_bypasses_quiet_hours(self, scheduler, mock_store, queue):
        """Alarms keep original urgency even during quiet hours."""
        now = datetime(2026, 3, 1, 2, 0)  # 2am = quiet hours
        mock_store.get_pending_reminders.return_value = [{
            "id": 3,
            "message": "Wake up!",
            "remind_at": now + timedelta(minutes=1),
            "lead_time": 5,
            "is_alarm": True,
            "urgency": 3,
            "recurring": None,
        }]

        scheduler._check_reminders(now)

        notif = queue.get_next_urgent()
        assert notif is not None
        assert notif.urgency == Urgency.ACTIVE

    def test_recurring_schedules_next(self, scheduler, mock_store, queue):
        """Recurring reminders should schedule next occurrence after delivery."""
        now = datetime(2026, 3, 1, 14, 0)
        reminder = {
            "id": 4,
            "message": "Daily standup",
            "remind_at": now + timedelta(minutes=1),
            "lead_time": 5,
            "is_alarm": False,
            "urgency": 2,
            "recurring": "daily",
        }
        mock_store.get_pending_reminders.return_value = [reminder]

        scheduler._check_reminders(now)

        mock_store.mark_reminder_delivered.assert_called_once_with(4)
        mock_store.schedule_next_recurrence.assert_called_once_with(reminder)


# ── Ambient Noise Gate ────────────────────────────────────────────────


class TestAmbientGate:
    def test_low_ambient_downgrades_gentle(self, mock_store, queue, mock_config):
        """Low ambient RMS should downgrade GENTLE → PASSIVE."""
        ambient_fn = MagicMock(return_value=0.001)  # very quiet
        sched = HeartbeatScheduler(
            memory_store=mock_store,
            notification_queue=queue,
            config=mock_config,
            ambient_fn=ambient_fn,
        )

        now = datetime(2026, 3, 1, 14, 0)  # not quiet hours
        mock_store.get_pending_reminders.return_value = [{
            "id": 5,
            "message": "Test",
            "remind_at": now + timedelta(minutes=1),
            "lead_time": 5,
            "is_alarm": False,
            "urgency": 2,  # GENTLE
            "recurring": None,
        }]

        sched._check_reminders(now)

        passive = queue.get_passive()
        assert len(passive) == 1
        assert passive[0].urgency == Urgency.PASSIVE

    def test_normal_ambient_keeps_urgency(self, mock_store, queue, mock_config):
        """Normal ambient RMS should keep original urgency."""
        ambient_fn = MagicMock(return_value=0.02)  # user present
        sched = HeartbeatScheduler(
            memory_store=mock_store,
            notification_queue=queue,
            config=mock_config,
            ambient_fn=ambient_fn,
        )

        now = datetime(2026, 3, 1, 14, 0)
        mock_store.get_pending_reminders.return_value = [{
            "id": 6,
            "message": "Test",
            "remind_at": now + timedelta(minutes=1),
            "lead_time": 5,
            "is_alarm": False,
            "urgency": 2,
            "recurring": None,
        }]

        sched._check_reminders(now)

        notif = queue.get_next_urgent()
        assert notif is not None
        assert notif.urgency == Urgency.GENTLE


# ── Built-in Rules ────────────────────────────────────────────────────


class TestBuiltinRules:
    def test_morning_greeting_fires_once(self, scheduler, queue):
        """Morning greeting should fire once between 06-10, not again."""
        now = datetime(2026, 3, 1, 7, 30)
        scheduler._last_reset_date = now.date()

        scheduler._check_builtin_rules(now)
        assert queue.size() == 1
        notif = queue.get_passive()
        assert len(notif) == 1
        assert notif[0].message == "__morning_greeting__"

        # Second call same day — should NOT fire again
        scheduler._check_builtin_rules(now)
        assert queue.is_empty()

    def test_morning_greeting_not_outside_hours(self, scheduler, queue):
        """Morning greeting should NOT fire outside 06-10."""
        scheduler._last_reset_date = datetime(2026, 3, 1).date()

        scheduler._check_builtin_rules(datetime(2026, 3, 1, 5, 59))
        assert queue.is_empty()

        scheduler._check_builtin_rules(datetime(2026, 3, 1, 10, 0))
        assert queue.is_empty()

    def test_sleep_reminder_fires_once(self, scheduler, queue, mock_config):
        """Sleep reminder at 23:00, once per night."""
        # Adjust quiet hours so 23 is NOT quiet (to allow GENTLE)
        mock_config.quiet_hours_start = 24  # effectively disabled
        now = datetime(2026, 3, 1, 23, 0)
        scheduler._last_reset_date = now.date()

        scheduler._check_builtin_rules(now)
        assert queue.size() == 1
        notif = queue.get_next_urgent()
        assert notif is not None
        assert notif.message == "__sleep_reminder__"
        assert notif.urgency == Urgency.GENTLE

        # Second call — should NOT fire again
        scheduler._check_builtin_rules(now)
        assert queue.is_empty()

    def test_morning_greeting_disabled(self, scheduler, queue, mock_config):
        """Morning greeting should not fire when disabled in config."""
        mock_config.morning_greeting_enabled = False
        scheduler._last_reset_date = datetime(2026, 3, 1).date()

        scheduler._check_builtin_rules(datetime(2026, 3, 1, 7, 30))
        assert queue.is_empty()

    def test_sleep_reminder_disabled(self, scheduler, queue, mock_config):
        """Sleep reminder should not fire when disabled in config."""
        mock_config.sleep_reminder_enabled = False
        mock_config.quiet_hours_start = 24
        scheduler._last_reset_date = datetime(2026, 3, 1).date()

        scheduler._check_builtin_rules(datetime(2026, 3, 1, 23, 0))
        assert queue.is_empty()


# ── Daily Flag Reset ──────────────────────────────────────────────────


class TestDailyReset:
    def test_flags_reset_at_midnight(self, scheduler, queue):
        """Daily flags should reset when date changes."""
        day1 = datetime(2026, 3, 1, 7, 0)
        scheduler._maybe_reset_daily_flags(day1)
        scheduler._morning_greeted = True
        scheduler._sleep_reminded = True

        # Next day
        day2 = datetime(2026, 3, 2, 7, 0)
        scheduler._maybe_reset_daily_flags(day2)
        assert scheduler._morning_greeted is False
        assert scheduler._sleep_reminded is False


# ── Quiet Hours ───────────────────────────────────────────────────────


class TestQuietHours:
    def test_is_quiet_late_night(self, scheduler):
        assert scheduler._is_quiet(datetime(2026, 3, 1, 23, 0)) is True
        assert scheduler._is_quiet(datetime(2026, 3, 1, 23, 59)) is True

    def test_is_quiet_early_morning(self, scheduler):
        assert scheduler._is_quiet(datetime(2026, 3, 1, 0, 0)) is True
        assert scheduler._is_quiet(datetime(2026, 3, 1, 5, 59)) is True

    def test_not_quiet_daytime(self, scheduler):
        assert scheduler._is_quiet(datetime(2026, 3, 1, 6, 0)) is False
        assert scheduler._is_quiet(datetime(2026, 3, 1, 14, 0)) is False
        assert scheduler._is_quiet(datetime(2026, 3, 1, 22, 59)) is False


# ── Tick Integration ──────────────────────────────────────────────────


class TestTick:
    def test_tick_calls_all_checks(self, scheduler, mock_store):
        """_tick should call reminders + rules + daily reset."""
        with patch.object(scheduler, "_check_reminders") as cr, \
             patch.object(scheduler, "_check_builtin_rules") as cbr, \
             patch.object(scheduler, "_maybe_reset_daily_flags") as mrd:
            scheduler._tick()
            cr.assert_called_once()
            cbr.assert_called_once()
            mrd.assert_called_once()


# ── Thread Lifecycle ──────────────────────────────────────────────────


class TestThreadLifecycle:
    def test_start_stop(self, scheduler, mock_config):
        """Scheduler should start and stop cleanly."""
        mock_config.heartbeat_interval = 0.1  # fast for testing
        scheduler.start()
        assert scheduler.is_running
        time.sleep(0.3)  # let a few ticks run
        scheduler.stop()
        assert not scheduler.is_running

    def test_double_start_no_crash(self, scheduler, mock_config):
        """Starting twice should not crash or create duplicate threads."""
        mock_config.heartbeat_interval = 0.1
        scheduler.start()
        scheduler.start()  # should log warning, not crash
        scheduler.stop()
