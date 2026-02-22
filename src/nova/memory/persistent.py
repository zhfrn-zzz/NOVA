"""Persistent user memory — stores facts about the user across sessions.

Facts are stored as key-value pairs in ~/.nova/memory.json and injected
into the LLM system prompt so NOVA remembers the user between restarts.
"""

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_MEMORY_DIR = Path.home() / ".nova"
_MEMORY_FILE = _MEMORY_DIR / "memory.json"

# Module-level singleton
_instance: "UserMemory | None" = None


class UserMemory:
    """Persistent key-value store for user facts.

    Loads from and saves to ~/.nova/memory.json automatically.
    """

    def __init__(self) -> None:
        self._facts: dict[str, str] = {}
        self._load()

    def _load(self) -> None:
        """Load facts from disk, creating the directory if needed."""
        if _MEMORY_FILE.exists():
            try:
                data = json.loads(_MEMORY_FILE.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    self._facts = {str(k): str(v) for k, v in data.items()}
                    logger.info(
                        "User memory loaded: %d facts from %s",
                        len(self._facts), _MEMORY_FILE,
                    )
            except (json.JSONDecodeError, OSError) as e:
                logger.warning("Failed to load user memory: %s", e)
        else:
            logger.debug("No user memory file found at %s", _MEMORY_FILE)

    def _save(self) -> None:
        """Persist facts to disk."""
        try:
            _MEMORY_DIR.mkdir(parents=True, exist_ok=True)
            _MEMORY_FILE.write_text(
                json.dumps(self._facts, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            logger.debug("User memory saved: %d facts", len(self._facts))
        except OSError as e:
            logger.error("Failed to save user memory: %s", e)

    def add_fact(self, key: str, value: str) -> None:
        """Store or update a fact about the user.

        Args:
            key: Fact identifier (e.g. "name", "location").
            value: Fact value (e.g. "Zhafran", "Bekasi").
        """
        key = key.strip().lower()
        value = value.strip()
        self._facts[key] = value
        self._save()
        logger.info("User memory: set %s=%s", key, value)

    def get_facts(self) -> dict[str, str]:
        """Return all stored facts as a dict copy.

        Returns:
            Dict of key-value fact pairs.
        """
        return dict(self._facts)

    def get_fact(self, key: str) -> str | None:
        """Return a single fact by key.

        Args:
            key: The fact key to look up.

        Returns:
            The fact value, or None if not found.
        """
        return self._facts.get(key.strip().lower())

    def remove_fact(self, key: str) -> bool:
        """Remove a fact by key.

        Args:
            key: The fact key to remove.

        Returns:
            True if the fact existed and was removed, False otherwise.
        """
        key = key.strip().lower()
        if key in self._facts:
            del self._facts[key]
            self._save()
            logger.info("User memory: removed %s", key)
            return True
        return False

    def clear(self) -> None:
        """Remove all stored facts."""
        self._facts.clear()
        self._save()
        logger.info("User memory cleared")

    @property
    def fact_count(self) -> int:
        """Number of facts currently stored."""
        return len(self._facts)


def get_user_memory() -> UserMemory:
    """Get the singleton UserMemory instance.

    Returns:
        The shared UserMemory loaded from disk.
    """
    global _instance
    if _instance is None:
        _instance = UserMemory()
    return _instance


# --- Tool functions called by the LLM via function calling ---

async def remember_fact(key: str, value: str) -> str:
    """Store a fact about the user in persistent memory.

    Args:
        key: Fact identifier (e.g. "name", "location", "hobby").
        value: Fact value.

    Returns:
        Confirmation message.
    """
    memory = get_user_memory()
    memory.add_fact(key, value)
    return f"Tersimpan: {key}={value}"


async def recall_facts() -> str:
    """Retrieve all stored facts about the user.

    Returns:
        Formatted string of all facts, or a message if none exist.
    """
    memory = get_user_memory()
    facts = memory.get_facts()
    if not facts:
        return "Belum ada informasi yang tersimpan tentang pengguna."
    lines = [f"{k}={v}" for k, v in facts.items()]
    return "User facts: " + ", ".join(lines)
