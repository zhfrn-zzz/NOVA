"""Conversation manager — persistent sliding window with auto-compaction.

Replaces the in-memory ConversationContext with a SQLite-backed manager
that survives restarts. When the window fills (20 turns), the oldest 15
are summarized via LLM, facts are extracted, and a daily log is written.
"""

import logging
from collections.abc import Awaitable, Callable
from datetime import datetime

logger = logging.getLogger(__name__)

# Type alias for the LLM summarization function
LLMFn = Callable[[str], Awaitable[str]]


class ConversationManager:
    """Persistent conversation history with auto-compaction.

    Stores turns in SQLite via MemoryStore, compacts old turns
    by summarizing them with the LLM, and recovers context on restart.
    """

    MAX_TURNS = 20
    COMPACT_AT = 15
    KEEP_RECENT = 5

    def __init__(
        self,
        memory_store: "MemoryStore",  # noqa: F821
        llm_fn: LLMFn | None = None,
    ) -> None:
        """Initialize the conversation manager.

        Args:
            memory_store: The MemoryStore instance for persistence.
            llm_fn: Async function that takes a prompt string and
                     returns a summary. Used for compaction.
                     If None, compaction will truncate without summarizing.
        """
        from nova.memory.memory_store import MemoryStore

        self._store: MemoryStore = memory_store
        self._llm_fn = llm_fn
        self._history: list[dict] = []
        self._session_id: int | None = None

        # Load recent turns from today on startup
        self._load_recent()

    def _load_recent(self) -> None:
        """On startup: load the last N turns from today's interactions."""
        recent = self._store.get_recent_interactions(
            limit=self.KEEP_RECENT * 2,
        )
        self._history = list(recent)
        if self._history:
            logger.info(
                "Loaded %d recent turns from database",
                len(self._history),
            )

    def start_session(self) -> None:
        """Start a new conversation session."""
        self._session_id = self._store.start_session()
        logger.info("Session started: #%s", self._session_id)

    def end_session(self, summary: str = "") -> None:
        """End the current session."""
        if self._session_id is not None:
            token_count = sum(
                len(t.get("content", "")) // 4 for t in self._history
            )
            self._store.end_session(
                self._session_id, summary=summary,
                token_count=token_count,
            )
            logger.info("Session ended: #%s", self._session_id)
            self._session_id = None

    async def add_turn(
        self,
        role: str,
        content: str,
        tool_calls: list | None = None,
    ) -> None:
        """Add a conversation turn and persist it.

        Triggers compaction if the window exceeds MAX_TURNS.

        Args:
            role: "user" or "assistant".
            content: Message content.
            tool_calls: Optional tool call data.
        """
        self._history.append({"role": role, "content": content})
        self._store.log_interaction(role, content, tool_calls)

        if len(self._history) >= self.MAX_TURNS * 2:
            await self._compact()

    async def add_exchange(
        self, user_msg: str, assistant_msg: str,
        tool_calls: list | None = None,
    ) -> None:
        """Add a user+assistant exchange pair.

        Convenience method that mirrors ConversationContext.add_exchange().

        Args:
            user_msg: The user's message.
            assistant_msg: The assistant's response.
            tool_calls: Optional tool calls made during this exchange.
        """
        await self.add_turn("user", user_msg)
        await self.add_turn("assistant", assistant_msg, tool_calls)

    def get_context(self) -> list[dict]:
        """Return the current conversation context.

        Returns:
            List of {"role": "user"|"assistant", "content": str} dicts.
        """
        return list(self._history)

    def clear(self) -> None:
        """Reset the conversation history."""
        self._history.clear()
        logger.debug("Conversation cleared")

    @property
    def turn_count(self) -> int:
        """Number of messages currently in the window."""
        return len(self._history)

    async def _compact(self) -> None:
        """Compact the conversation by summarizing old turns.

        Steps:
        1. Summarize the oldest COMPACT_AT*2 messages via LLM
        2. Write summary to the daily log
        3. Auto-extract facts from the conversation
        4. Replace old messages with a summary bridge
        """
        compact_count = self.COMPACT_AT * 2
        old = self._history[:compact_count]
        recent = self._history[compact_count:]

        text = "\n".join(
            f"{t['role']}: {t['content']}" for t in old
        )

        # Summarize via LLM
        summary = ""
        if self._llm_fn:
            try:
                summary = await self._llm_fn(
                    "Summarize this conversation concisely. "
                    "Key facts, decisions, requests. "
                    "Indonesian. Max 100 words.\n\n" + text,
                )
            except Exception:
                logger.warning(
                    "Compaction summarization failed", exc_info=True,
                )
                # Fallback: use last few messages as summary
                summary = "Percakapan sebelumnya berisi " + ", ".join(
                    t["content"][:50] for t in old[-4:]
                )

            # Auto-extract facts
            try:
                await self._extract_facts(text)
            except Exception:
                logger.warning(
                    "Fact extraction failed", exc_info=True,
                )
        else:
            summary = "Percakapan sebelumnya berisi " + ", ".join(
                t["content"][:50] for t in old[-4:]
            )

        # Write daily log
        self._append_daily_log(summary)

        # Replace history with summary bridge + recent turns
        self._history = [
            {
                "role": "system",
                "content": (
                    f"[Ringkasan percakapan sebelumnya: {summary}]"
                ),
            },
            *recent,
        ]

        logger.info(
            "Compacted %d turns → summary (%d chars) + %d recent",
            len(old), len(summary), len(recent),
        )

    async def _extract_facts(self, text: str) -> None:
        """Extract and store long-term facts from conversation text."""
        if not self._llm_fn:
            return

        result = await self._llm_fn(
            "Extract important long-term facts about the user "
            "from this conversation. "
            "key: value format, one per line. "
            "If nothing worth remembering, output NONE.\n\n" + text,
        )

        if result.strip().upper() == "NONE":
            return

        for line in result.strip().split("\n"):
            if ":" in line:
                k, v = line.split(":", 1)
                k = k.strip()
                v = v.strip()
                if k and v:
                    self._store.store_memory(k, v, source="auto")
                    logger.info("Auto-extracted fact: %s=%s", k, v)

    def _append_daily_log(self, summary: str) -> None:
        """Append a compaction summary to today's daily log file."""
        daily_dir = self._store._db_path.parent / "daily"
        daily_dir.mkdir(parents=True, exist_ok=True)

        today = datetime.now().strftime("%Y-%m-%d")
        log_path = daily_dir / f"{today}.md"

        try:
            timestamp = datetime.now().strftime("%H:%M")
            entry = f"\n## {timestamp}\n\n{summary}\n"

            if log_path.exists():
                current = log_path.read_text(encoding="utf-8")
                log_path.write_text(
                    current + entry, encoding="utf-8",
                )
            else:
                header = f"# Daily Log — {today}\n"
                log_path.write_text(
                    header + entry, encoding="utf-8",
                )
            logger.debug("Daily log updated: %s", log_path)
        except OSError as e:
            logger.warning("Failed to write daily log: %s", e)
