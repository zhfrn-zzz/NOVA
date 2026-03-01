"""Notification queue for NOVA's heartbeat system.

Thread-safe queue with three urgency levels for managing proactive
notifications: passive (deliver on next interaction), gentle (chime + listen),
and active (alert + speak immediately).
"""

import threading
from dataclasses import dataclass
from datetime import datetime
from enum import IntEnum


class Urgency(IntEnum):
    """Notification urgency levels."""

    PASSIVE = 1   # deliver on next user interaction
    GENTLE = 2    # play chime, wait for user to respond
    ACTIVE = 3    # play alert + speak immediately


@dataclass
class Notification:
    """A single notification in the heartbeat queue."""

    message: str
    urgency: Urgency
    source: str              # "reminder" | "rule" | "system"
    created_at: datetime
    reminder_id: int | None = None
    attempts: int = 0
    max_attempts: int = 3


class NotificationQueue:
    """Thread-safe notification queue for heartbeat notifications."""

    def __init__(self) -> None:
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
        """Check if the queue is empty."""
        with self._lock:
            return len(self._queue) == 0

    def size(self) -> int:
        """Return the number of notifications in the queue."""
        with self._lock:
            return len(self._queue)

    def clear(self) -> None:
        """Remove all notifications from the queue."""
        with self._lock:
            self._queue.clear()
