# Prompt 2: Heartbeat Scheduler + Audio

Read CLAUDE.md. Implement the heartbeat scheduler and notification audio.
Prerequisite: Task 39 (reminders table, queue, tools) must be complete.

## 1. Create src/nova/heartbeat/scheduler.py

HeartbeatScheduler class with:

- Background daemon thread, checks every `config.heartbeat_interval` seconds (default 60)
- `start()` / `stop()` methods
- `_tick()` called every interval:
  1. Reset daily flags at midnight
  2. Check pending reminders via memory_store.get_pending_reminders()
  3. Check built-in rules (morning greeting, sleep reminder)
  4. Push notifications to queue with appropriate urgency

### Reminder checking logic:
```python
def _check_reminders(self, now):
    pending = self._store.get_pending_reminders(now, window_minutes=2)
    for r in pending:
        urgency = Urgency(r["urgency"])
        
        # Quiet hours: downgrade non-alarm to PASSIVE
        if self._is_quiet(now) and not r["is_alarm"]:
            urgency = Urgency.PASSIVE
        
        # Ambient gate: if very quiet, downgrade GENTLE to PASSIVE
        if self._ambient_fn and urgency >= Urgency.GENTLE:
            if self._ambient_fn() < config.ambient_presence_threshold:
                urgency = Urgency.PASSIVE
        
        self._queue.push(Notification(
            message=r["message"],
            urgency=urgency,
            source="reminder",
            created_at=now,
            reminder_id=r["id"],
        ))
        self._store.mark_reminder_delivered(r["id"])
        
        # Handle recurring
        if r.get("recurring"):
            self._store.schedule_next_recurrence(r)
```

### Built-in rules:
- Morning greeting: PASSIVE, once per day, between 06:00-09:59
  Message: `"__morning_greeting__"` (special token, orchestrator interprets)
- Sleep reminder: GENTLE, once per night, at 23:00
  Message: `"__sleep_reminder__"`
- Daily flags reset at midnight

### Quiet hours:
- Default 23:00 - 06:00 (from config)
- During quiet hours: all non-alarm notifications downgraded to PASSIVE
- `_is_quiet(now)`: returns True if hour >= quiet_start or hour < quiet_end

### Ambient noise gate:
- Constructor accepts optional `ambient_fn: Callable[[], float]`
- This function returns current ambient RMS from wake word detector
- If ambient < threshold for current check, downgrade GENTLE → PASSIVE
- Does NOT suppress ACTIVE+is_alarm (alarms always fire)

## 2. Create src/nova/heartbeat/audio.py

Two functions that generate audio as numpy int16 arrays:

**generate_chime(sample_rate=22050) -> np.ndarray**
- Gentle two-note ascending chime (C5→E5)
- Duration: 0.6 seconds
- Volume: controlled by config.chime_volume
- Smooth envelope (fade in/out)

**generate_alert(sample_rate=22050) -> np.ndarray**
- Urgent alternating two-tone (A5/E5)
- Duration: 1.0 seconds
- Volume: controlled by config.alert_volume
- More aggressive envelope

**play_notification_sound(audio: np.ndarray, sample_rate=22050) -> None**
- Play audio array through default output device
- Use `sounddevice` if available, otherwise fallback to writing temp wav + system play
- Block until playback complete
- Handle errors gracefully (no crash if audio device busy)

## 3. Update config.py

Add these fields to NovaConfig:

```python
# Heartbeat
heartbeat_enabled: bool = True
heartbeat_interval: int = 60
quiet_hours_start: int = 23
quiet_hours_end: int = 6
morning_greeting_enabled: bool = True
sleep_reminder_enabled: bool = True
ambient_presence_threshold: float = 0.005

# Notification audio
chime_volume: float = 0.3
alert_volume: float = 0.5
gentle_listen_timeout: int = 5
gentle_max_retries: int = 3
gentle_retry_delay: int = 300
```

## 4. Tests in tests/test_heartbeat_scheduler.py

Test (use mocked time, mocked store, mocked queue):
- _tick() with pending reminder → pushes to queue
- _tick() during quiet hours → downgrades non-alarm to PASSIVE
- _tick() with alarm during quiet hours → keeps original urgency
- Morning greeting: fires once between 06:00-09:59, not again same day
- Sleep reminder: fires at 23:00, not again same night
- Daily flags reset at midnight
- Ambient gate: low ambient → downgrade GENTLE to PASSIVE
- Recurring reminder: after delivery, next occurrence scheduled
- Scheduler start/stop (thread management)

Test audio (tests/test_heartbeat_audio.py):
- generate_chime returns correct shape and dtype
- generate_alert returns correct shape and dtype
- Audio values within int16 range

## Verification Checklist

- [ ] HeartbeatScheduler starts background thread on start()
- [ ] Stops cleanly on stop() without hanging
- [ ] Pending reminders pushed to queue with correct urgency
- [ ] Quiet hours downgrade works
- [ ] Alarm bypasses quiet hours
- [ ] Morning greeting fires once per day
- [ ] Sleep reminder fires once per night
- [ ] Daily flags reset at midnight
- [ ] Ambient noise gate downgrades when quiet
- [ ] generate_chime() produces valid audio array
- [ ] generate_alert() produces valid audio array
- [ ] `python -m pytest tests/ -x` — all pass
- [ ] `ruff check src/ tests/` — no NEW errors
- [ ] CLAUDE.md updated with Task 40

Do NOT modify orchestrator.py yet — that's Prompt 3.
