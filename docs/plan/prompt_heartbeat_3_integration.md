# Prompt 3: Heartbeat Orchestrator Integration

Read CLAUDE.md. Wire the heartbeat system into NOVA's main loop and orchestrator.
Prerequisites: Task 39 (data layer) and Task 40 (scheduler + audio) must be complete.

## 1. Wire heartbeat into orchestrator.__init__()

In Orchestrator.__init__():
- Import HeartbeatScheduler, NotificationQueue from nova.heartbeat
- Create NotificationQueue instance
- Create HeartbeatScheduler with memory_store, queue, and ambient_fn
  - ambient_fn: get from wake word detector if available (the ambient RMS value)
  - If wake word detector not available (text-only mode), pass None
- If config.heartbeat_enabled: call scheduler.start()
- On shutdown: call scheduler.stop()

## 2. Passive notification delivery (Level 1)

In the orchestrator method that handles user interactions (both text and voice),
BEFORE the LLM call:

```python
# Check for passive notifications
passive_notes = self._notification_queue.get_passive()
if passive_notes:
    # Format for LLM context
    note_text = self._format_notifications(passive_notes)
    # Inject into system prompt via prompt assembler
    # Add to memory_context or a separate notification_context
```

Create `_format_notifications(notifications) -> str`:
- For "__morning_greeting__": "Deliver a brief morning greeting to the user."
- For "__sleep_reminder__": "Gently remind the user it's late and they should rest."
- For reminder messages: "Remind the user: {message}"
- Return formatted string for LLM context injection

Update prompt_assembler to support notification context:
- Add `set_notification_context(context: str)` method (similar to set_memory_context)
- In build(): append notification context section if present
- Section header: "Pending notifications to deliver (incorporate naturally):"

## 3. Gentle notification delivery (Level 2)

This requires modifying the main wake word listener loop.

In the main loop (likely in main.py or wherever the wake word loop runs):

```python
# In the main wake word listening loop, between iterations:

if notification_queue.has_urgent():
    notif = notification_queue.get_next_urgent()
    
    if notif.urgency == Urgency.GENTLE:
        # 1. Pause wake word detector
        wake_detector.stop()
        
        # 2. Play chime
        from nova.heartbeat.audio import generate_chime, play_notification_sound
        chime = generate_chime()
        play_notification_sound(chime)
        
        # 3. Listen for user response (short timeout)
        audio = audio_capture.listen(timeout=config.gentle_listen_timeout)
        
        if audio is not None:
            # User responded — process as normal interaction
            # with notification in context
            text = await stt.transcribe(audio)
            # Add notification to context for this interaction
            orchestrator.set_pending_notification(notif)
            response = await orchestrator.handle_voice_interaction(text, ...)
        else:
            # User didn't respond
            notif.attempts += 1
            if notif.attempts < notif.max_attempts:
                # Re-queue for later retry
                # (scheduler will re-check on next tick, or use a delayed re-queue)
                notification_queue.push(notif)
            else:
                # Max retries — downgrade to passive
                notif.urgency = Urgency.PASSIVE
                notification_queue.push(notif)
        
        # 4. Resume wake word detector
        wake_detector.start()
```

## 4. Active notification delivery (Level 3)

Same location as gentle, but instead of just chime:

```python
    elif notif.urgency == Urgency.ACTIVE:
        # 1. Pause wake word detector
        wake_detector.stop()
        
        # 2. Play alert sound
        from nova.heartbeat.audio import generate_alert, play_notification_sound
        alert = generate_alert()
        play_notification_sound(alert)
        
        # 3. Generate and speak notification via LLM + TTS
        # Use orchestrator to generate a natural reminder message
        prompt = f"Urgent reminder for the user: {notif.message}. Deliver it concisely."
        # Use the streaming path to generate and speak
        await orchestrator.deliver_notification(notif)
        
        # 4. Brief listen for user response
        audio = audio_capture.listen(timeout=config.gentle_listen_timeout)
        if audio is not None:
            text = await stt.transcribe(audio)
            await orchestrator.handle_voice_interaction(text, ...)
        
        # 5. Resume wake word detector
        wake_detector.start()
```

## 5. Add deliver_notification() to orchestrator

```python
async def deliver_notification(self, notification: Notification) -> None:
    """Generate and speak a notification message."""
    # Build a minimal LLM prompt for the notification
    if notification.message == "__morning_greeting__":
        prompt = "Deliver a brief, warm morning greeting."
    elif notification.message == "__sleep_reminder__":
        prompt = "Gently remind the user it's late and time to rest."
    else:
        prompt = f"Deliver this reminder concisely: {notification.message}"
    
    # Use LLM to generate natural wording
    context = []  # No conversation history needed for notifications
    async for sentence in self._llm.generate_stream(prompt, context):
        await self._tts.speak(sentence)
```

## 6. Expose ambient RMS from wake word detector

The wake word detector tracks ambient noise levels. Expose this:
- Add a method `get_ambient_rms() -> float` to the wake word detector class
- This returns the current ambient RMS value
- Pass this as ambient_fn to HeartbeatScheduler

Find where the wake word detector is instantiated and how it tracks
ambient RMS (look for ambient_rms, _ambient_rms, or similar).
Create a simple wrapper function that the scheduler can call.

## 7. Handle text-only mode

In --text-only mode:
- Heartbeat scheduler still runs (for reminders)
- GENTLE/ACTIVE: print to console instead of audio
  "[NOVA notification] Pak, ujian Anda 5 menit lagi."
- No chime/alert sounds (no audio device needed)
- Passive: still inject into LLM context on next input

## 8. Tests in tests/test_heartbeat_integration.py

Test:
- Passive notifications injected into LLM context
- format_notifications handles special tokens correctly
- Gentle flow: chime → listen → timeout → retry
- Active flow: alert → TTS → listen
- Text-only mode: print instead of audio
- Scheduler stop on orchestrator shutdown
- Notification context appears in assembled prompt

## Verification Checklist

- [ ] Heartbeat scheduler starts on NOVA startup
- [ ] Heartbeat scheduler stops cleanly on exit
- [ ] Passive notifications injected into LLM context on next interaction
- [ ] LLM naturally incorporates morning greeting / reminders in response
- [ ] Gentle: chime plays, NOVA listens for response
- [ ] Active: alert plays, NOVA speaks reminder
- [ ] Text-only mode: notifications print to console
- [ ] Ambient RMS exposed from wake word detector
- [ ] `python -m pytest tests/ -x` — all pass
- [ ] `ruff check src/ tests/` — no NEW errors
- [ ] CLAUDE.md updated with Task 41
- [ ] Manual test: set_reminder 2 minutes from now → notification fires
