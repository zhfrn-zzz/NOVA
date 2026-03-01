"""Heartbeat-aware reminder tools — ISO 8601 datetime-based reminders.

Replaces the old minutes-based set_reminder with proper datetime-based
reminders stored in SQLite. Used by the heartbeat scheduler for proactive
notification delivery.
"""

import logging
from datetime import datetime

from nova.memory.memory_store import get_memory_store

logger = logging.getLogger(__name__)


async def set_reminder(
    message: str,
    remind_at: str,
    lead_time: int = 5,
    is_alarm: bool = False,
    recurring: str | None = None,
) -> str:
    """Set a reminder at a specific datetime.

    Args:
        message: Reminder text.
        remind_at: ISO 8601 datetime, e.g. "2026-03-02T08:00:00".
        lead_time: Minutes before remind_at to notify (default 5).
        is_alarm: If True, bypasses quiet hours.
        recurring: null | "daily" | "weekly" | "weekdays".

    Returns:
        Confirmation message.
    """
    try:
        dt = datetime.fromisoformat(remind_at)
    except ValueError:
        return (
            f"Format waktu tidak valid: {remind_at}. "
            "Gunakan ISO 8601, contoh: 2026-03-02T08:00:00"
        )

    if dt < datetime.now():
        return f"Waktu reminder sudah lewat: {remind_at}"

    store = get_memory_store()
    rid = store.add_reminder(
        message=message,
        remind_at=remind_at,
        lead_time=lead_time,
        is_alarm=is_alarm,
        recurring=recurring,
    )

    formatted = dt.strftime("%d %b %Y %H:%M")
    result = f"Reminder #{rid} diset: '{message}' pada {formatted}"
    if lead_time > 0:
        result += f" (notifikasi {lead_time} menit sebelumnya)"
    if recurring:
        result += f" [recurring: {recurring}]"

    logger.info("Tool set_reminder → #%d, at=%s, msg=%r", rid, remind_at, message)
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
