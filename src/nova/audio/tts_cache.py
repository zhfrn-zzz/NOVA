"""Simple in-memory LRU cache for TTS audio bytes.

Caches ``(text, voice, language) → audio_bytes`` to avoid redundant
TTS API calls for repeated phrases (e.g. tool confirmations).

Memory-only, no disk persistence. Bounded by both entry count and
total byte size to stay within NOVA's strict RAM budget.
"""

import hashlib
import logging
from collections import OrderedDict

logger = logging.getLogger(__name__)


class TTSCache:
    """LRU cache for TTS audio bytes.

    Args:
        max_entries: Maximum number of cached entries.
        max_bytes: Maximum total size of cached audio in bytes.
    """

    def __init__(
        self, max_entries: int = 50, max_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self._cache: OrderedDict[str, bytes] = OrderedDict()
        self._max_entries = max_entries
        self._max_bytes = max_bytes
        self._current_bytes = 0

    @staticmethod
    def _key(text: str, voice: str, language: str) -> str:
        raw = f"{text}|{voice}|{language}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get(self, text: str, voice: str, language: str) -> bytes | None:
        """Look up cached audio.

        Returns audio bytes on hit, None on miss. Promotes the entry
        to most-recently-used on hit.
        """
        key = self._key(text, voice, language)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        return None

    def put(self, text: str, voice: str, language: str, audio: bytes) -> None:
        """Store audio bytes in the cache, evicting LRU entries as needed."""
        key = self._key(text, voice, language)
        if key in self._cache:
            self._current_bytes -= len(self._cache[key])
            del self._cache[key]

        # Evict until we have space
        while (
            len(self._cache) >= self._max_entries
            or self._current_bytes + len(audio) > self._max_bytes
        ):
            if not self._cache:
                break
            _, evicted = self._cache.popitem(last=False)
            self._current_bytes -= len(evicted)

        self._cache[key] = audio
        self._current_bytes += len(audio)

    @property
    def size(self) -> int:
        """Number of entries currently in the cache."""
        return len(self._cache)

    @property
    def bytes_used(self) -> int:
        """Total bytes of audio currently cached."""
        return self._current_bytes


# Module-level singleton so the cache is shared across the TTS provider.
_instance: TTSCache | None = None


def get_tts_cache() -> TTSCache:
    """Return the global TTSCache singleton."""
    global _instance
    if _instance is None:
        _instance = TTSCache()
    return _instance
