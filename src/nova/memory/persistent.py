"""Persistent user memory — backward-compatible wrapper over MemoryStore.

Maintains the same public API (remember_fact, recall_facts) for the tool
registry while delegating to the new SQLite-backed MemoryStore.
The old ~/.nova/memory.json is automatically migrated on first access.
"""

import logging

logger = logging.getLogger(__name__)


# --- Legacy-compatible UserMemory wrapper ---


class UserMemory:
    """Persistent key-value store for user facts.

    Now delegates to MemoryStore (SQLite) instead of flat JSON.
    """

    def __init__(self) -> None:
        from nova.memory.memory_store import get_memory_store

        self._store = get_memory_store()

    def add_fact(self, key: str, value: str) -> None:
        """Store or update a fact about the user."""
        self._store.store_memory(key, value, source="user")

    def get_facts(self) -> dict[str, str]:
        """Return all stored facts as a dict copy."""
        return self._store.get_all_memories()

    def get_fact(self, key: str) -> str | None:
        """Return a single fact by key."""
        return self._store.get_memory(key)

    def remove_fact(self, key: str) -> bool:
        """Remove a fact by key."""
        return self._store.delete_memory(key)

    def clear(self) -> None:
        """Remove all stored facts."""
        for key in list(self._store.get_all_memories().keys()):
            self._store.delete_memory(key)
        logger.info("User memory cleared")

    @property
    def fact_count(self) -> int:
        """Number of facts currently stored."""
        return self._store.memory_count()


# Module-level singleton
_instance: "UserMemory | None" = None


def get_user_memory() -> UserMemory:
    """Get the singleton UserMemory instance.

    Returns:
        The shared UserMemory (backed by SQLite).
    """
    global _instance
    if _instance is None:
        _instance = UserMemory()
    return _instance


def reset_user_memory() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None


# --- Tool functions called by the LLM via function calling ---


async def memory_store(key: str, value: str) -> str:
    """Store a fact about the user in persistent memory.

    Args:
        key: Fact identifier (e.g. "name", "location", "hobby").
        value: Fact value.

    Returns:
        Confirmation message.
    """
    from nova.memory.memory_store import get_memory_store

    get_memory_store().store_memory(key, value, source="user")
    return f"Tersimpan: {key}={value}"


async def memory_search(query: str) -> str:
    """Search stored memories by relevance.

    Args:
        query: Search query to find relevant memories.

    Returns:
        Formatted string of matching memories.
    """
    from nova.memory.memory_store import get_memory_store
    from nova.memory.retriever import MemoryRetriever

    store = get_memory_store()
    retriever = MemoryRetriever(memory_store=store)
    results = await retriever.search(query)

    if not results:
        return "Tidak ada memori yang relevan ditemukan."

    lines = [f"{r['key']}={r['value']}" for r in results]
    return "Memori relevan: " + ", ".join(lines)


async def memory_forget(key: str) -> str:
    """Remove a specific memory by key.

    Args:
        key: The fact key to forget.

    Returns:
        Confirmation or not-found message.
    """
    from nova.memory.memory_store import get_memory_store

    if get_memory_store().delete_memory(key):
        return f"Terhapus: {key}"
    return f"Memori '{key}' tidak ditemukan."


async def update_user_profile(info: str) -> str:
    """Add information to the user profile (USER.md).

    Args:
        info: Text to append to the user profile.

    Returns:
        Confirmation message.
    """
    from nova.memory.prompt_assembler import get_prompt_assembler

    get_prompt_assembler().update_user_profile(info)
    return f"Profil diperbarui: {info}"


# --- Backward-compatible aliases ---

async def remember_fact(key: str, value: str) -> str:
    """Legacy alias for memory_store."""
    return await memory_store(key, value)


async def recall_facts() -> str:
    """Legacy alias — returns all stored facts."""
    from nova.memory.memory_store import get_memory_store

    facts = get_memory_store().get_all_memories()
    if not facts:
        return "Belum ada informasi yang tersimpan tentang pengguna."
    lines = [f"{k}={v}" for k, v in facts.items()]
    return "User facts: " + ", ".join(lines)
