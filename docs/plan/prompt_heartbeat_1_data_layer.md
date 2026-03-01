# Prompt 1: Heartbeat Data Layer

Read CLAUDE.md. Implement the data layer for NOVA's heartbeat system:
reminders table, LLM tools, and notification queue.

## 1. Add reminders table to memory_store.py

Add to the _SCHEMA string (after sessions table):

```sql
CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message     TEXT NOT NULL,
    remind_at   TEXT NOT NULL,
    lead_time   INTEGER DEFAULT 5,
    is_alarm    BOOLEAN DEFAULT 0,
    urgency     INTEGER DEFAULT 2,
    recurring   TEXT,
    delivered   BOOLEAN DEFAULT 0,
    created_at  TEXT NOT NULL,
    delivered_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reminders_pending
    ON reminders(remind_at) WHERE delivered = 0;
```

Add these methods to MemoryStore class:

- `add_reminder(message, remind_at, lead_time=5, is_alarm=False, urgency=2, recurring=None) -> int`
  Returns reminder id. `remind_at` is ISO 8601 string.

- `get_pending_reminders(now: datetime, window_minutes: int = 2) -> list[dict]`
  Returns reminders where: not delivered AND remind_at minus lead_time <= now + window.
  Each dict: id, message, remind_at (as datetime), lead_time, is_alarm, urgency, recurring.

- `mark_reminder_delivered(reminder_id: int) -> None`
  Set delivered=1, delivered_at=now.

- `cancel_reminder(reminder_id: int) -> bool`
  Delete reminder by id. Return True if found and deleted.

- `list_reminders(include_delivered=False) -> list[dict]`
  Return all reminders. If include_delivered=False, only pending ones.

- `schedule_next_recurrence(reminder: dict) -> int | None`
  If reminder has recurring field:
    "daily" → add 1 day to remind_at
    "weekly" → add 7 days
    "weekdays" → add 1 day, skip sat/sun
  Create new reminder with updated remind_at. Return new id or None.

## 2. Create src/nova/heartbeat/__init__.py

Empty file.

## 3. Create src/nova/heartbeat/queue.py

```python
"""Notification queue for NOVA's heartbeat system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import IntEnum
import threading


class Urgency(IntEnum):
    PASSIVE = 1   # deliver on next user interaction
    GENTLE = 2    # play chime, wait for user to respond
    ACTIVE = 3    # play alert + speak immediately


@dataclass
class Notification:
    message: str
    urgency: Urgency
    source: str              # "reminder" | "rule" | "system"
    created_at: datetime
    reminder_id: int | None = None
    attempts: int = 0
    max_attempts: int = 3


class NotificationQueue:
    """Thread-safe notification queue for heartbeat notifications."""

    def __init__(self):
        self._queue: list[Notification] = []
        self._lock = threading.Lock()

    def push(self, notification: Notification) -> None:
        """Add a notification to the queue."""
        with self._lock:
            self._queue.append(notification)

    def get_passive(self) -> list[Notification]:
        """Get and remove all PASSIVE notifications."""
        with self._lock:
            passive = [n for n in self._queue if n.urgency == Urgency.PASSIVE]
            self._queue = [n for n in self._queue if n.urgency != Urgency.PASSIVE]
            return passive

    def get_next_urgent(self) -> Notification | None:
        """Get and remove the highest urgency GENTLE/ACTIVE notification."""
        with self._lock:
            urgent = [n for n in self._queue if n.urgency >= Urgency.GENTLE]
            if not urgent:
                return None
            urgent.sort(key=lambda n: (-n.urgency, n.created_at))
            notif = urgent[0]
            self._queue.remove(notif)
            return notif

    def has_urgent(self) -> bool:
        """Check if there are any GENTLE or ACTIVE notifications."""
        with self._lock:
            return any(n.urgency >= Urgency.GENTLE for n in self._queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        with self._lock:
            self._queue.clear()
```

## 4. Add reminder tools to registry.py

Add 3 new tools:

**set_reminder** — Parameters: message (str), remind_at (str, ISO 8601 or natural language that LLM converts to ISO), lead_time (int, default 5), is_alarm (bool, default false), recurring (str|null, one of "daily"/"weekly"/"weekdays"/null).

The implementation should:
- Parse remind_at as ISO 8601 datetime
- Call memory_store.add_reminder()
- Return confirmation string with formatted time

**list_reminders** — No parameters. Returns formatted list of pending reminders.

**cancel_reminder** — Parameters: reminder_id (int). Cancels the reminder.

Also register these tools in the TOOLS list with proper Gemini function declarations.

## 5. Update RULES.md

Add this section to ~/.nova/prompts/RULES.md (via prompt_assembler default content):

```
Reminders vs Memory:
- "ingatkan saya besok jam 8 ada ujian" → set_reminder (has specific time)
- "jam 3 sore ada meeting" → set_reminder (has specific time)
- "ingat saya suka kopi" → memory_store (fact, no time)
- "saya kerja di Wantimpres" → memory_store (fact, no time)
When setting reminders, convert relative times to absolute ISO 8601 datetime.
"besok jam 8" with current date 2026-03-01 → "2026-03-02T08:00:00".
"30 menit lagi" with current time 10:00 → "2026-03-01T10:30:00".
```

Update the default RULES.md content in prompt_assembler.py to include this.

## 6. Tests in tests/test_heartbeat_data.py

Test:
- add_reminder and get_pending_reminders
- mark_reminder_delivered (should not appear in pending)
- cancel_reminder
- list_reminders (with and without delivered)
- schedule_next_recurrence for daily, weekly, weekdays
- NotificationQueue: push, get_passive, get_next_urgent, has_urgent
- NotificationQueue: thread safety (push from multiple threads)
- NotificationQueue: get_next_urgent returns highest urgency first

## Verification Checklist

- [ ] `reminders` table created on startup (check nova.db)
- [ ] set_reminder tool registered and callable
- [ ] list_reminders returns pending reminders
- [ ] cancel_reminder removes reminder from DB
- [ ] schedule_next_recurrence creates correct next occurrence
- [ ] RULES.md updated with reminder guidance
- [ ] NotificationQueue is thread-safe
- [ ] `python -m pytest tests/ -x` — all pass
- [ ] `ruff check src/ tests/` — no NEW errors
- [ ] CLAUDE.md updated with Task 39

Do NOT implement the scheduler or audio yet — that's Prompt 2.
