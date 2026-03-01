# NOVA Heartbeat System — Design Document

## Overview

Heartbeat membuat NOVA proactive — bisa ingatkan jadwal, sapa di pagi hari,
dan notify user tanpa diminta. Tiga level urgency dengan notification queue
memastikan NOVA tidak mengganggu di waktu yang salah.

---

## Component 1: Reminders Table

### Why separate from memories?

Memories: `ujian_besok: Ujian jam 8 pagi` — tidak ada datetime yang parseable.
Reminders: `message: "Ujian", remind_at: "2026-03-02T08:00:00"` — exact datetime.

Scanner cukup `SELECT * FROM reminders WHERE remind_at <= now AND NOT delivered`.
Tidak perlu NLP, tidak perlu LLM call.

### Schema (tambah di nova.db)

```sql
CREATE TABLE IF NOT EXISTS reminders (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    message     TEXT NOT NULL,
    remind_at   TEXT NOT NULL,           -- ISO 8601: "2026-03-02T08:00:00"
    lead_time   INTEGER DEFAULT 5,       -- notify X minutes BEFORE remind_at
    is_alarm    BOOLEAN DEFAULT 0,       -- if true, bypasses quiet hours
    urgency     INTEGER DEFAULT 2,       -- 1=passive, 2=gentle, 3=active
    recurring   TEXT,                    -- null | "daily" | "weekly" | "weekdays"
    delivered   BOOLEAN DEFAULT 0,
    created_at  TEXT NOT NULL,
    delivered_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_reminders_pending
    ON reminders(remind_at) WHERE delivered = 0;
```

### LLM Tools

```
set_reminder(message, remind_at, lead_time=5, is_alarm=false, recurring=null)
  → "Reminder set: Ujian at 2026-03-02 08:00 (notify 5 min before)"

list_reminders()
  → "1. Ujian — 2026-03-02 08:00 (5 min before)\n2. Meeting — ..."

cancel_reminder(id)
  → "Reminder #1 cancelled"
```

Key behavior: ketika user bilang "ingatkan saya besok ujian jam 8",
LLM harus panggil set_reminder, BUKAN memory_store. Ini perlu
instruksi eksplisit di RULES.md:

```
When user asks to be reminded of something with a specific time,
use set_reminder (not memory_store). Examples:
- "ingatkan saya besok jam 8 ada ujian" → set_reminder
- "ingat saya suka kopi" → memory_store (no time, just a fact)
```

---

## Component 2: Notification Queue

### Data Structures

```python
from enum import IntEnum
from dataclasses import dataclass, field
from datetime import datetime
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
    attempts: int = 0        # how many times chime played (for GENTLE)
    max_attempts: int = 3    # stop retrying after this


class NotificationQueue:
    """Thread-safe notification queue."""

    def __init__(self):
        self._queue: list[Notification] = []
        self._lock = threading.Lock()

    def push(self, notification: Notification) -> None:
        with self._lock:
            self._queue.append(notification)

    def get_passive(self) -> list[Notification]:
        """Get and remove all passive notifications."""
        with self._lock:
            passive = [n for n in self._queue if n.urgency == Urgency.PASSIVE]
            self._queue = [n for n in self._queue if n.urgency != Urgency.PASSIVE]
            return passive

    def get_next_urgent(self) -> Notification | None:
        """Get highest urgency GENTLE/ACTIVE notification."""
        with self._lock:
            urgent = [n for n in self._queue if n.urgency >= Urgency.GENTLE]
            if not urgent:
                return None
            urgent.sort(key=lambda n: (-n.urgency, n.created_at))
            notif = urgent[0]
            self._queue.remove(notif)
            return notif

    def has_urgent(self) -> bool:
        with self._lock:
            return any(n.urgency >= Urgency.GENTLE for n in self._queue)

    def is_empty(self) -> bool:
        with self._lock:
            return len(self._queue) == 0
```

---

## Component 3: Heartbeat Scheduler

### Logic

```python
class HeartbeatScheduler:
    """Background thread that checks reminders and rules every 60 seconds."""

    CHECK_INTERVAL = 60      # seconds
    QUIET_START = 23         # 23:00
    QUIET_END = 6            # 06:00
    REMINDER_SCAN_WINDOW = 2 # check reminders due within 2 minutes

    def __init__(self, memory_store, notification_queue, ambient_fn=None):
        self._store = memory_store
        self._queue = notification_queue
        self._ambient_fn = ambient_fn    # fn() -> float (current ambient RMS)
        self._stop_event = threading.Event()
        self._thread = None

        # Daily flags
        self._morning_greeted = False
        self._sleep_reminded = False
        self._last_reset_date = None

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        logger.info("Heartbeat scheduler started (interval=%ds)", self.CHECK_INTERVAL)

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _run(self):
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.exception("Heartbeat tick failed")
            self._stop_event.wait(self.CHECK_INTERVAL)

    def _tick(self):
        now = datetime.now()
        self._maybe_reset_daily_flags(now)
        self._check_reminders(now)
        self._check_builtin_rules(now)

    # ── Reminders ──

    def _check_reminders(self, now):
        """Find reminders that are due and push to queue."""
        pending = self._store.get_pending_reminders(now, self.REMINDER_SCAN_WINDOW)

        for reminder in pending:
            # Calculate effective notify time
            notify_at = reminder["remind_at"] - timedelta(minutes=reminder["lead_time"])

            if now >= notify_at:
                urgency = Urgency(reminder["urgency"])

                # Quiet hours check
                if self._is_quiet(now) and not reminder["is_alarm"]:
                    # Downgrade to passive during quiet hours
                    urgency = Urgency.PASSIVE

                # Ambient noise gate (Tier 1 presence)
                if self._ambient_fn and urgency >= Urgency.GENTLE:
                    ambient = self._ambient_fn()
                    if ambient < 0.005:  # very quiet = likely away/sleeping
                        urgency = Urgency.PASSIVE

                self._queue.push(Notification(
                    message=reminder["message"],
                    urgency=urgency,
                    source="reminder",
                    created_at=now,
                    reminder_id=reminder["id"],
                ))

                # Mark as delivered
                self._store.mark_reminder_delivered(reminder["id"])

                # Handle recurring
                if reminder.get("recurring"):
                    self._store.schedule_next_recurrence(reminder)

                logger.info(
                    "Reminder triggered: '%s' (urgency=%s)",
                    reminder["message"], urgency.name,
                )

    # ── Built-in Rules ──

    def _check_builtin_rules(self, now):
        hour = now.hour

        # Morning greeting (once per day, 06:00-09:59)
        if 6 <= hour < 10 and not self._morning_greeted:
            self._queue.push(Notification(
                message="__morning_greeting__",
                urgency=Urgency.PASSIVE,
                source="rule",
                created_at=now,
            ))
            self._morning_greeted = True
            logger.debug("Morning greeting queued")

        # Sleep reminder (23:00, once per night)
        if hour == 23 and not self._sleep_reminded:
            if not self._is_quiet(now):
                # Only remind if not already in quiet mode
                self._queue.push(Notification(
                    message="__sleep_reminder__",
                    urgency=Urgency.GENTLE,
                    source="rule",
                    created_at=now,
                ))
                self._sleep_reminded = True
                logger.debug("Sleep reminder queued")

    # ── Helpers ──

    def _is_quiet(self, now):
        hour = now.hour
        return hour >= self.QUIET_START or hour < self.QUIET_END

    def _maybe_reset_daily_flags(self, now):
        today = now.date()
        if self._last_reset_date != today:
            self._morning_greeted = False
            self._sleep_reminded = False
            self._last_reset_date = today
```

---

## Component 4: Audio Notifications

### Sound Generation (no external files needed)

```python
import numpy as np

def generate_chime(sample_rate=22050) -> np.ndarray:
    """Gentle two-note chime for GENTLE notifications."""
    duration = 0.6  # seconds
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Two ascending notes (C5 → E5)
    note1 = np.sin(2 * np.pi * 523 * t[:len(t)//2]) * 0.3
    note2 = np.sin(2 * np.pi * 659 * t[len(t)//2:]) * 0.3

    # Envelope (fade in/out)
    envelope = np.concatenate([
        np.linspace(0, 1, len(t)//4),
        np.ones(len(t)//4),
        np.linspace(1, 0, len(t)//2),
    ])[:len(t)]

    audio = np.concatenate([note1, note2]) * envelope
    return (audio * 32767).astype(np.int16)


def generate_alert(sample_rate=22050) -> np.ndarray:
    """Urgent two-tone alert for ACTIVE notifications."""
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration))

    # Alternating tones (urgent feel)
    freq = np.where(
        (t * 4).astype(int) % 2 == 0,
        880,  # A5
        660,  # E5
    )
    audio = np.sin(2 * np.pi * freq * t) * 0.5

    envelope = np.concatenate([
        np.linspace(0, 1, len(t)//8),
        np.ones(len(t) * 6 // 8),
        np.linspace(1, 0, len(t)//8),
    ])[:len(t)]

    audio = (audio * envelope * 32767).astype(np.int16)
    return audio
```

### Playback Integration

NOVA sudah punya audio output untuk TTS. Chime/alert pakai channel yang sama.
Sebelum play: pause wake word detector (supaya chime tidak trigger wake word).
Setelah play: resume.

---

## Component 5: Orchestrator Integration

### Passive Delivery (Level 1)

```
User: "hey nova, jam berapa?"

Orchestrator:
  1. Check notification queue → passive items exist
  2. Format: "Pending notifications: morning_greeting"
  3. Add to LLM context: "You have a pending morning greeting to deliver.
     Incorporate it naturally into your response."
  4. LLM generates: "Selamat pagi, Pak. Sekarang pukul 8 lewat 15.
     Oh ya, hari ini diprediksi cerah di Bekasi."
```

### Gentle Delivery (Level 2)

```
Main loop (between wake word iterations):
  1. Check queue → has_urgent() == True
  2. Get notification → urgency == GENTLE
  3. Pause wake word detector
  4. Play chime sound
  5. Enter "listening for response" mode (5 seconds)
  6. If user speaks → process as normal interaction
     (notification message added to LLM context)
  7. If silence → resume wake word detector
     notification.attempts += 1
     if attempts < max_attempts: re-queue with retry delay
     else: downgrade to PASSIVE
```

### Active Delivery (Level 3)

```
Main loop:
  1. Check queue → urgency == ACTIVE
  2. Pause wake word detector
  3. Play alert sound
  4. Generate TTS via LLM:
     Context: "Urgent reminder: Ujian 5 menit lagi"
     LLM: "Pak, ujian Anda 5 menit lagi."
  5. Play TTS
  6. Enter "listening" mode briefly
  7. Resume wake word detector
```

### Integration Flow

```
┌─────────────────────────────────────────────────────┐
│                   MAIN LOOP                         │
│                                                     │
│  while running:                                     │
│    ├─ Check notification queue                      │
│    │   ├─ ACTIVE? → pause ww → alert → TTS → listen│
│    │   ├─ GENTLE? → pause ww → chime → listen       │
│    │   └─ nothing? → continue                       │
│    │                                                │
│    ├─ Wake word detected?                           │
│    │   ├─ Yes → capture audio → STT                 │
│    │   │        → check passive queue               │
│    │   │        → inject to context                 │
│    │   │        → LLM stream → TTS                  │
│    │   └─ No → continue                             │
│    │                                                │
│    └─ sleep(wake_word_frame_duration)               │
└─────────────────────────────────────────────────────┘
```

---

## Component 6: Ambient Noise Gate (Tier 1 Presence)

NOVA's wake word detector already tracks ambient RMS.
Expose this as a function the heartbeat scheduler can call.

```
ambient_rms < 0.005 for 30+ minutes
  → likely: user sleeping or not in room
  → action: downgrade GENTLE→PASSIVE, suppress ACTIVE (unless alarm)

ambient_rms > 0.01
  → user probably present (typing, moving, breathing picked up)
  → action: normal urgency levels apply
```

This is NOT presence detection — it's absence heuristic.
False positives (user is present but very still) are safe:
GENTLE becomes PASSIVE, user gets notification on next interaction.
False negatives (user left but room is noisy) are rare and harmless.

---

## Configuration (config.py additions)

```python
# Heartbeat
heartbeat_enabled: bool = True
heartbeat_interval: int = 60              # seconds between checks
quiet_hours_start: int = 23               # 23:00
quiet_hours_end: int = 6                  # 06:00
morning_greeting_enabled: bool = True
sleep_reminder_enabled: bool = True
ambient_presence_threshold: float = 0.005 # RMS below = likely away

# Notification
chime_volume: float = 0.3                 # 0.0-1.0
alert_volume: float = 0.5
gentle_listen_timeout: int = 5            # seconds to wait after chime
gentle_max_retries: int = 3
gentle_retry_delay: int = 300             # seconds between retries (5 min)
```

---

## RULES.md Addition

```
Reminders:
- When user asks to be reminded of something at a specific time, use set_reminder.
- When user states a fact without time reference, use memory_store.
- Examples:
  "ingatkan saya besok jam 8 ada ujian" → set_reminder
  "ingat saya suka kopi" → memory_store
  "jam 3 sore ada meeting" → set_reminder
  "saya kerja di Wantimpres" → memory_store

Notifications:
- When you have pending notifications, incorporate them naturally.
- Morning greeting: be warm but brief. "Selamat pagi, Pak."
- Don't list all notifications mechanically — weave them into conversation.
```

---

## Implementation Plan (3 prompts)

### Prompt 1: Data Layer (reminders table + tools + queue)
- Add reminders table to memory_store.py schema
- CRUD methods: add_reminder, get_pending_reminders, mark_delivered, cancel, list
- set_reminder / list_reminders / cancel_reminder tools in registry.py
- NotificationQueue class in new file: src/nova/heartbeat/queue.py
- Update RULES.md with reminder vs memory_store guidance
- Tests for all CRUD + queue operations

### Prompt 2: Scheduler + Audio
- HeartbeatScheduler in src/nova/heartbeat/scheduler.py
- Background thread, 60s tick, reminder scanning, built-in rules
- Quiet hours logic, daily flag reset
- Chime/alert audio generation in src/nova/heartbeat/audio.py
- Ambient noise gate integration (get ambient from wake word detector)
- Tests for scheduler logic (mock time, mock store)

### Prompt 3: Orchestrator Integration
- Wire heartbeat scheduler start/stop in orchestrator
- Passive queue check before LLM calls (inject to context)
- Gentle notification: chime → listen → process or retry
- Active notification: alert → LLM generate → TTS → listen
- Main loop modification to check queue between wake word frames
- Integration tests

---

## File Structure

```
src/nova/heartbeat/
├── __init__.py
├── queue.py          # NotificationQueue, Notification, Urgency
├── scheduler.py      # HeartbeatScheduler (background thread)
└── audio.py          # generate_chime(), generate_alert()
```

---

## Token/API Budget Impact

```
Heartbeat checks: 0 API calls (pure datetime comparison)
Built-in rules:   0 API calls (hardcoded logic)
Active delivery:  1 LLM call (generate TTS text for reminder)
Passive delivery:  0 extra calls (piggyback on user's interaction)

Worst case: 5 active reminders/day = 5 extra LLM calls
            + 120 regular interactions = 125 total
            Well within 1500 RPD free tier.
```

---

## Summary

Heartbeat mengubah NOVA dari reactive ke proactive dengan overhead minimal:
zero API calls untuk checking, notification queue dengan 3 urgency levels,
quiet hours protection, dan ambient noise gate sebagai heuristic kehadiran.
User tetap in control — NOVA tidak pernah interruptive kecuali benar-benar urgent.
