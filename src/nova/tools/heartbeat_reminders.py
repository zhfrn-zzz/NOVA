"""Heartbeat-aware reminder tools — ISO 8601 datetime-based reminders.

Supports both absolute times (remind_at) and relative times (delay_minutes).
Stored in SQLite. Used by the heartbeat scheduler for proactive notification
delivery.
"""

import logging
from datetime import datetime, timedelta

from nova.memory.memory_store import get_memory_store

logger = logging.getLogger(__name__)


async def set_reminder(
    message: str,
    remind_at: str | None = None,
    delay_minutes: int | None = None,
    lead_time: int = 5,
    is_alarm: bool = False,
    recurring: str | None = None,
    action: dict | None = None,
) -> str:
    """Set a reminder at a specific datetime or after a delay.

    Args:
        message: Reminder text.
        remind_at: ISO 8601 datetime, e.g. "2026-03-02T08:00:00".
                   Optional if delay_minutes is provided.
        delay_minutes: Minutes from now. If provided, overrides remind_at.
        lead_time: Minutes before remind_at to notify (default 5).
        is_alarm: If True, bypasses quiet hours.
        recurring: null | "daily" | "weekly" | "weekdays".
        action: Optional IoT action to execute automatically when reminder fires.
                Dict with keys: device, action, value (optional).

    Returns:
        Confirmation message.
    """
    # --- Resolve remind_at ---------------------------------------------------
    if delay_minutes is not None:
        # Auto-clamp lead_time: if delay is shorter than lead_time,
        # notify at exact remind_at (lead_time=0), not before it was created
        if delay_minutes <= lead_time:
            lead_time = 0

        # Relative time: Python calculates the exact datetime
        dt = datetime.now() + timedelta(minutes=delay_minutes)
        remind_at = dt.isoformat(timespec="seconds")
    elif remind_at is not None:
        # Absolute time: validate and check not in the past
        try:
            dt = datetime.fromisoformat(remind_at)
        except ValueError:
            return (
                f"Format waktu tidak valid: {remind_at}. "
                "Gunakan ISO 8601, contoh: 2026-03-02T08:00:00"
            )
        if dt < datetime.now():
            return f"Waktu reminder sudah lewat: {remind_at}"
    else:
        return "Harus isi remind_at atau delay_minutes."

    store = get_memory_store()
    rid = store.add_reminder(
        message=message,
        remind_at=remind_at,
        lead_time=lead_time,
        is_alarm=is_alarm,
        recurring=recurring,
        action=action,
    )

    formatted = dt.strftime("%d %b %Y %H:%M")
    result = f"Reminder #{rid} diset: '{message}' pada {formatted}"
    if lead_time > 0:
        result += f" (notifikasi {lead_time} menit sebelumnya)"
    if recurring:
        result += f" [recurring: {recurring}]"
    if action:
        cmd = action.get("command") or action.get("action", "?")
        result += f" (akan otomatis eksekusi: {action['device']} → {cmd})"

    logger.info(
        "Tool set_reminder → #%d, at=%s, msg=%r, action=%r",
        rid, remind_at, message, action,
    )
    return result


async def list_reminders() -> str:
    """List all pending reminders.

    Returns:
        Formatted list of pending reminders, or 'no reminders' message.
    """
    store = get_memory_store()
    reminders = store.list_reminders(include_delivered=False)

    if not reminders:
        return "Tidak ada reminder yang aktif."

    lines = []
    for r in reminders:
        try:
            dt = datetime.fromisoformat(r["remind_at"])
            formatted = dt.strftime("%d %b %Y %H:%M")
        except ValueError:
            formatted = r["remind_at"]

        line = f"#{r['id']}. {r['message']} — {formatted}"
        if r.get("recurring"):
            line += f" [{r['recurring']}]"
        lines.append(line)

    return "\n".join(lines)


async def cancel_reminder(reminder_id: int) -> str:
    """Cancel a reminder by ID.

    Args:
        reminder_id: The reminder ID to cancel.

    Returns:
        Confirmation or error message.
    """
    store = get_memory_store()
    if store.cancel_reminder(reminder_id):
        return f"Reminder #{reminder_id} dibatalkan."
    return f"Reminder #{reminder_id} tidak ditemukan."
