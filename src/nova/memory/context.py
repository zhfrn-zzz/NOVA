"""Sliding window conversation context manager."""

import logging

logger = logging.getLogger(__name__)


class ConversationContext:
    """Manages conversation history with a sliding window.

    Keeps the last N exchanges (user + assistant pairs) to provide
    context to the LLM without unbounded memory growth.
    """

    def __init__(self, max_turns: int = 10) -> None:
        """Initialize the context manager.

        Args:
            max_turns: Maximum number of exchanges to retain.
                       Each exchange = one user message + one assistant message.
        """
        self._history: list[dict] = []
        self._max_turns = max_turns

    def add_exchange(self, user_msg: str, assistant_msg: str) -> None:
        """Add a user/assistant exchange and trim if needed.

        Args:
            user_msg: The user's message.
            assistant_msg: The assistant's response.
        """
        self._history.append({"role": "user", "content": user_msg})
        self._history.append({"role": "assistant", "content": assistant_msg})

        # Each exchange is 2 messages, so max messages = max_turns * 2
        max_messages = self._max_turns * 2
        if len(self._history) > max_messages:
            overflow = len(self._history) - max_messages
            self._history = self._history[overflow:]
            logger.debug(
                "Context trimmed: removed %d messages, %d remaining",
                overflow, len(self._history),
            )

    def get_context(self) -> list[dict]:
        """Return the current conversation context.

        Returns:
            List of {"role": "user"|"assistant", "content": str} dicts.
        """
        return list(self._history)

    def clear(self) -> None:
        """Reset the conversation context."""
        self._history.clear()
        logger.debug("Context cleared")

    @property
    def turn_count(self) -> int:
        """Number of exchanges currently stored."""
        return len(self._history) // 2
