"""Persistent user memory — tool functions for the LLM via function calling.

Delegates to the SQLite-backed MemoryStore for all storage operations.
"""


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
