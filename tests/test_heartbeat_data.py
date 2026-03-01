"""Tests for heartbeat data layer — reminders CRUD + notification queue."""

import threading
from datetime import datetime, timedelta

import pytest

from nova.heartbeat.queue import Notification, NotificationQueue, Urgency
from nova.memory.memory_store import MemoryStore


@pytest.fixture
def store(tmp_path):
    """Create a MemoryStore with a temporary database."""
    db = tmp_path / "test_heartbeat.db"
    s = MemoryStore(db_path=db)
    yield s
    s.close()


# ── Reminder CRUD ────────────────────────────────────────────────────


class TestAddReminder:
    def test_add_and_list(self, store):
        rid = store.add_reminder("Ujian", "2026-03-02T08:00:00")
        assert rid > 0
        reminders = store.list_reminders()
        assert len(reminders) == 1
        assert reminders[0]["message"] == "Ujian"
        assert reminders[0]["remind_at"] == "2026-03-02T08:00:00"

    def test_add_with_all_params(self, store):
        store.add_reminder(
            message="Meeting",
            remind_at="2026-03-02T15:00:00",
            lead_time=10,
            is_alarm=True,
            urgency=3,
            recurring="daily",
        )
        reminders = store.list_reminders()
        assert len(reminders) == 1
        r = reminders[0]
        assert r["lead_time"] == 10
        assert r["is_alarm"] is True
        assert r["urgency"] == 3
        assert r["recurring"] == "daily"


class TestGetPendingReminders:
    def test_due_reminder_appears(self, store):
        # Reminder at now + 3 min, lead_time=5 → effective notify = now - 2 min → due
        now = datetime.now()
        remind_at = now + timedelta(minutes=3)
        store.add_reminder("Due soon", remind_at.isoformat(), lead_time=5)

        pending = store.get_pending_reminders(now, window_minutes=2)
        assert len(pending) == 1
        assert pending[0]["message"] == "Due soon"
        assert isinstance(pending[0]["remind_at"], datetime)

    def test_future_reminder_not_pending(self, store):
        # Reminder far in the future → not due
        future = datetime.now() + timedelta(hours=24)
        store.add_reminder("Future", future.isoformat(), lead_time=5)

        pending = store.get_pending_reminders(datetime.now(), window_minutes=2)
        assert len(pending) == 0


class TestMarkDelivered:
    def test_delivered_not_in_pending(self, store):
        now = datetime.now()
        remind_at = now + timedelta(minutes=1)
        rid = store.add_reminder("Test", remind_at.isoformat(), lead_time=5)

        store.mark_reminder_delivered(rid)

        pending = store.get_pending_reminders(now, window_minutes=10)
        assert len(pending) == 0

    def test_delivered_excluded_from_list(self, store):
        rid = store.add_reminder("Test", "2026-03-02T08:00:00")
        store.mark_reminder_delivered(rid)

        assert len(store.list_reminders(include_delivered=False)) == 0
        assert len(store.list_reminders(include_delivered=True)) == 1


class TestCancelReminder:
    def test_cancel_existing(self, store):
        rid = store.add_reminder("Cancel me", "2026-03-02T08:00:00")
        assert store.cancel_reminder(rid) is True
        assert len(store.list_reminders()) == 0

    def test_cancel_nonexistent(self, store):
        assert store.cancel_reminder(9999) is False


class TestListReminders:
    def test_empty_list(self, store):
        assert store.list_reminders() == []

    def test_list_pending_only(self, store):
        rid1 = store.add_reminder("Pending", "2026-03-02T08:00:00")
        rid2 = store.add_reminder("Delivered", "2026-03-01T08:00:00")
        store.mark_reminder_delivered(rid2)

        pending = store.list_reminders(include_delivered=False)
        assert len(pending) == 1
        assert pending[0]["id"] == rid1

    def test_list_all(self, store):
        store.add_reminder("A", "2026-03-02T08:00:00")
        rid2 = store.add_reminder("B", "2026-03-01T08:00:00")
        store.mark_reminder_delivered(rid2)

        all_reminders = store.list_reminders(include_delivered=True)
        assert len(all_reminders) == 2


class TestScheduleNextRecurrence:
    def test_daily(self, store):
        store.add_reminder("Daily", "2026-03-01T08:00:00", recurring="daily")
        reminder = store.list_reminders()[0]
        new_rid = store.schedule_next_recurrence(reminder)
        assert new_rid is not None

        reminders = store.list_reminders()
        assert len(reminders) == 2
        next_r = [r for r in reminders if r["id"] == new_rid][0]
        assert next_r["remind_at"] == "2026-03-02T08:00:00"

    def test_weekly(self, store):
        store.add_reminder("Weekly", "2026-03-01T08:00:00", recurring="weekly")
        reminder = store.list_reminders()[0]
        new_rid = store.schedule_next_recurrence(reminder)
        assert new_rid is not None

        reminders = store.list_reminders()
        next_r = [r for r in reminders if r["id"] == new_rid][0]
        assert next_r["remind_at"] == "2026-03-08T08:00:00"

    def test_weekdays_skips_weekend(self, store):
        # 2026-03-06 is Friday → next weekday should be Monday 2026-03-09
        store.add_reminder("Weekday", "2026-03-06T08:00:00", recurring="weekdays")
        reminder = store.list_reminders()[0]
        new_rid = store.schedule_next_recurrence(reminder)
        assert new_rid is not None

        reminders = store.list_reminders()
        next_r = [r for r in reminders if r["id"] == new_rid][0]
        next_dt = datetime.fromisoformat(next_r["remind_at"])
        assert next_dt.weekday() == 0  # Monday
        assert next_r["remind_at"] == "2026-03-09T08:00:00"

    def test_no_recurrence(self, store):
        store.add_reminder("One-time", "2026-03-01T08:00:00")
        reminder = store.list_reminders()[0]
        result = store.schedule_next_recurrence(reminder)
        assert result is None


# ── NotificationQueue ────────────────────────────────────────────────


class TestNotificationQueue:
    def test_push_and_size(self):
        q = NotificationQueue()
        assert q.is_empty()
        q.push(Notification("hi", Urgency.PASSIVE, "test", datetime.now()))
        assert q.size() == 1
        assert not q.is_empty()

    def test_get_passive(self):
        q = NotificationQueue()
        q.push(Notification("p1", Urgency.PASSIVE, "test", datetime.now()))
        q.push(Notification("g1", Urgency.GENTLE, "test", datetime.now()))
        q.push(Notification("p2", Urgency.PASSIVE, "test", datetime.now()))

        passive = q.get_passive()
        assert len(passive) == 2
        assert all(n.urgency == Urgency.PASSIVE for n in passive)
        # GENTLE should still be in queue
        assert q.size() == 1

    def test_get_next_urgent_returns_highest_first(self):
        q = NotificationQueue()
        now = datetime.now()
        q.push(Notification("gentle", Urgency.GENTLE, "test", now))
        q.push(Notification("active", Urgency.ACTIVE, "test", now))
        q.push(Notification("passive", Urgency.PASSIVE, "test", now))

        notif = q.get_next_urgent()
        assert notif is not None
        assert notif.message == "active"
        assert notif.urgency == Urgency.ACTIVE
        # 2 left (gentle + passive)
        assert q.size() == 2

    def test_get_next_urgent_empty(self):
        q = NotificationQueue()
        q.push(Notification("p", Urgency.PASSIVE, "test", datetime.now()))
        assert q.get_next_urgent() is None

    def test_has_urgent(self):
        q = NotificationQueue()
        q.push(Notification("p", Urgency.PASSIVE, "test", datetime.now()))
        assert q.has_urgent() is False
        q.push(Notification("g", Urgency.GENTLE, "test", datetime.now()))
        assert q.has_urgent() is True

    def test_clear(self):
        q = NotificationQueue()
        q.push(Notification("a", Urgency.ACTIVE, "test", datetime.now()))
        q.push(Notification("b", Urgency.GENTLE, "test", datetime.now()))
        q.clear()
        assert q.is_empty()

    def test_thread_safety(self):
        """Push from multiple threads and verify all items arrive."""
        q = NotificationQueue()
        count = 100

        def push_batch(start):
            for i in range(start, start + count):
                q.push(Notification(
                    f"msg-{i}", Urgency.PASSIVE, "test", datetime.now(),
                ))

        threads = [threading.Thread(target=push_batch, args=(i * count,))
                   for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert q.size() == 4 * count

    def test_urgency_ordering_same_level(self):
        """When multiple GENTLE notifications, oldest first."""
        q = NotificationQueue()
        t1 = datetime(2026, 1, 1, 10, 0)
        t2 = datetime(2026, 1, 1, 11, 0)
        q.push(Notification("later", Urgency.GENTLE, "test", t2))
        q.push(Notification("earlier", Urgency.GENTLE, "test", t1))

        notif = q.get_next_urgent()
        assert notif is not None
        assert notif.message == "earlier"
