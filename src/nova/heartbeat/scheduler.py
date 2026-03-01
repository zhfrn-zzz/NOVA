"""Heartbeat scheduler — background thread for proactive notifications.

Checks reminders and built-in rules every `heartbeat_interval` seconds.
Pushes notifications to the queue with appropriate urgency, respecting
quiet hours and ambient noise gate heuristics.
"""

import logging
import threading
from collections.abc import Callable
from datetime import date, datetime

from nova.config import NovaConfig, get_config
from nova.heartbeat.queue import Notification, NotificationQueue, Urgency

logger = logging.getLogger(__name__)


class HeartbeatScheduler:
    """Background thread that checks reminders and rules periodically."""

    def __init__(
        self,
        memory_store: object,
        notification_queue: NotificationQueue,
        config: NovaConfig | None = None,
        ambient_fn: Callable[[], float] | None = None,
    ) -> None:
        """Initialize the heartbeat scheduler.

        Args:
            memory_store: MemoryStore instance with reminder CRUD methods.
            notification_queue: Queue to push notifications into.
            config: NovaConfig instance (uses singleton if None).
            ambient_fn: Optional callable returning current ambient RMS.
        """
        self._store = memory_store
        self._queue = notification_queue
        self._config = config or get_config()
        self._ambient_fn = ambient_fn
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Daily flags
        self._morning_greeted = False
        self._sleep_reminded = False
        self._last_reset_date: date | None = None

    def start(self) -> None:
        """Start the background heartbeat thread."""
        if self._thread and self._thread.is_alive():
            logger.warning("Heartbeat scheduler already running")
            return

        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info(
            "Heartbeat scheduler started (interval=%ds)",
            self._config.heartbeat_interval,
        )

    def stop(self) -> None:
        """Stop the background heartbeat thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None
        logger.info("Heartbeat scheduler stopped")

    @property
    def is_running(self) -> bool:
        """Check if the scheduler thread is alive."""
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        """Main loop — tick every interval until stopped."""
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Heartbeat tick failed")
            self._stop_event.wait(self._config.heartbeat_interval)

    def _tick(self) -> None:
        """Single heartbeat iteration."""
        now = datetime.now()
        self._maybe_reset_daily_flags(now)
        self._check_reminders(now)
        self._check_builtin_rules(now)

    # ── Reminders ──────────────────────────────────────────────────

    def _check_reminders(self, now: datetime) -> None:
        """Find due reminders and push to notification queue."""
        pending = self._store.get_pending_reminders(now, window_minutes=2)

        for r in pending:
            urgency = Urgency(r["urgency"])

            # Quiet hours: downgrade non-alarm to PASSIVE
            if self._is_quiet(now) and not r["is_alarm"]:
                urgency = Urgency.PASSIVE

            # Ambient noise gate: very quiet → likely away/sleeping
            if self._ambient_fn and urgency >= Urgency.GENTLE:
                ambient = self._ambient_fn()
                if ambient < self._config.ambient_presence_threshold:
                    urgency = Urgency.PASSIVE

            self._queue.push(Notification(
                message=r["message"],
                urgency=urgency,
                source="reminder",
                created_at=now,
                reminder_id=r["id"],
            ))

            # Mark as delivered
            self._store.mark_reminder_delivered(r["id"])

            # Handle recurring
            if r.get("recurring"):
                self._store.schedule_next_recurrence(r)

            logger.info(
                "Reminder triggered: '%s' (urgency=%s)",
                r["message"], urgency.name,
            )

    # ── Built-in Rules ─────────────────────────────────────────────

    def _check_builtin_rules(self, now: datetime) -> None:
        """Check time-based rules: morning greeting, sleep reminder."""
        hour = now.hour

        # Morning greeting (once per day, 06:00-09:59)
        if (
            self._config.morning_greeting_enabled
            and 6 <= hour < 10
            and not self._morning_greeted
        ):
            self._queue.push(Notification(
                message="__morning_greeting__",
                urgency=Urgency.PASSIVE,
                source="rule",
                created_at=now,
            ))
            self._morning_greeted = True
            logger.debug("Morning greeting queued")

        # Sleep reminder (23:00, once per night)
        if (
            self._config.sleep_reminder_enabled
            and hour == 23
            and not self._sleep_reminded
        ):
            if not self._is_quiet(now):
                # Only remind if not already in quiet mode
                # (quiet hours start at 23 by default, but this fires
                #  at the boundary hour before quiet check takes effect
                #  for non-alarm items)
                self._queue.push(Notification(
                    message="__sleep_reminder__",
                    urgency=Urgency.GENTLE,
                    source="rule",
                    created_at=now,
                ))
            self._sleep_reminded = True
            logger.debug("Sleep reminder queued")

    # ── Helpers ────────────────────────────────────────────────────

    def _is_quiet(self, now: datetime) -> bool:
        """Check if current time is within quiet hours."""
        hour = now.hour
        return hour >= self._config.quiet_hours_start or hour < self._config.quiet_hours_end

    def _maybe_reset_daily_flags(self, now: datetime) -> None:
        """Reset daily flags at midnight."""
        today = now.date()
        if self._last_reset_date != today:
            self._morning_greeted = False
            self._sleep_reminded = False
            self._last_reset_date = today
