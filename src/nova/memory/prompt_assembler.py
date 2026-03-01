"""File-based prompt assembler — builds system prompts from editable Markdown files.

Loads personality (SOUL.md), rules (RULES.md), and user profile (USER.md) from
~/.nova/prompts/. Creates default files on first run. Supports hot-reload
by checking file mtime — no restart needed to change NOVA's personality.
"""

import datetime
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
_BULAN = [
    "", "Januari", "Februari", "Maret", "April", "Mei", "Juni",
    "Juli", "Agustus", "September", "Oktober", "November", "Desember",
]

# --- Default prompt file content ---

_DEFAULT_SOUL = """\
You are NOVA, a personal AI assistant created by Zhafran.
You are modeled after JARVIS — calm, composed, and quietly competent.
You speak with a refined, slightly formal tone but never stiff or robotic.

Address the user as "Sir" or "Pak" naturally.
Be efficient and precise — deliver information, not filler.
Show quiet confidence. State things, don't hedge.
Light sarcasm is acceptable when appropriate, but always respectful.
You are loyal and proactive.
"""

_DEFAULT_RULES = """\
Response rules:
- Keep responses between 20-50 words unless user asks for detail.
- Responses will be spoken aloud — plain text only.
- No markdown, bullet points, asterisks, emoji, exclamation marks.
- Default to Indonesian unless user speaks English.
- Never start with "Tentu" or "Baik" — answer or act directly.

Tool usage:
- Use tools immediately. Don't ask confirmation unless destructive.
- Only call web_search once per question.
- Answer from search results directly, never say "saya menemukan hasil."
- When the user shares personal information, use memory_store to save it.
- When the user asks if you remember something, use memory_search to check.

Reminders vs Memory:
- "ingatkan saya besok jam 8 ada ujian" → set_reminder (has specific time)
- "jam 3 sore ada meeting" → set_reminder (has specific time)
- "ingat saya suka kopi" → memory_store (fact, no time)
- "saya kerja di Wantimpres" → memory_store (fact, no time)
When setting reminders, convert relative times to absolute ISO 8601 datetime.
"besok jam 8" with current date 2026-03-01 → "2026-03-02T08:00:00".
"30 menit lagi" with current time 10:00 → "2026-03-01T10:30:00".
"""

_DEFAULT_USER = """\
Name: Zhafran
Location: Bekasi, Indonesia
Occupation: 11th grade vocational student, Computer Network Engineering
Currently interning at Wantimpres
Interests: AI, men's fashion (old money/Victorian), fragrances
"""

# Module-level singleton
_instance: "PromptAssembler | None" = None


class PromptAssembler:
    """Assembles system prompt from file components + dynamic context.

    Reads SOUL.md, RULES.md, USER.md from the prompts directory. Creates
    default files on first access. Caches file contents and reloads only
    when the file's mtime changes (hot-reload without restart).
    """

    def __init__(self, prompts_dir: str | Path | None = None) -> None:
        """Initialize the assembler.

        Args:
            prompts_dir: Path to the prompts directory.
                         Defaults to ~/.nova/prompts/.
        """
        if prompts_dir is None:
            self._dir = Path.home() / ".nova" / "prompts"
        else:
            self._dir = Path(prompts_dir)

        # Cache: filename -> (mtime, content)
        self._cache: dict[str, tuple[float, str]] = {}

        # Pending memory context (set by orchestrator, consumed by build())
        self._pending_memory_context: str = ""

        # Pending notification context (set by orchestrator, consumed by build())
        self._pending_notification_context: str = ""

        # Ensure directory and defaults exist
        self._ensure_defaults()

    def _ensure_defaults(self) -> None:
        """Create the prompts directory and default files if they don't exist."""
        self._dir.mkdir(parents=True, exist_ok=True)

        defaults = {
            "SOUL.md": _DEFAULT_SOUL,
            "RULES.md": _DEFAULT_RULES,
            "USER.md": _DEFAULT_USER,
        }

        for filename, content in defaults.items():
            path = self._dir / filename
            if not path.exists():
                path.write_text(content, encoding="utf-8")
                logger.info("Created default prompt file: %s", path)

    def _read_cached(self, filename: str) -> str:
        """Read a prompt file, using cache if mtime hasn't changed.

        Args:
            filename: Name of the file in the prompts directory.

        Returns:
            File content as string, or empty string if file doesn't exist.
        """
        path = self._dir / filename
        if not path.exists():
            return ""

        try:
            mtime = path.stat().st_mtime
            cached = self._cache.get(filename)

            if cached and cached[0] == mtime:
                return cached[1]

            content = path.read_text(encoding="utf-8").strip()
            self._cache[filename] = (mtime, content)
            return content
        except OSError as e:
            logger.warning("Failed to read prompt file %s: %s", filename, e)
            return ""

    def build(self, memory_context: str = "", datetime_str: str = "") -> str:
        """Assemble the final system prompt. Called every LLM request.

        Args:
            memory_context: Relevant memories from hybrid search.
            datetime_str: Current datetime string. Auto-generated if empty.

        Returns:
            Complete system prompt string.
        """
        sections: list[str] = []

        # 1. Core identity (SOUL.md)
        soul = self._read_cached("SOUL.md")
        if soul:
            sections.append(soul)

        # 2. Rules (RULES.md)
        rules = self._read_cached("RULES.md")
        if rules:
            sections.append(rules)

        # 3. User profile (USER.md)
        user = self._read_cached("USER.md")
        if user:
            sections.append(f"About the user:\n{user}")

        # 4. Current datetime — injected every call
        if not datetime_str:
            datetime_str = self._get_datetime_str()
        sections.append(f"Current date and time: {datetime_str}")

        # 5. Relevant memories — from hybrid search, NOT full dump
        if not memory_context:
            memory_context = self._pending_memory_context
            self._pending_memory_context = ""  # Consume once
        if memory_context:
            sections.append(f"Relevant memories:\n{memory_context}")

        # 6. Notification context — pending heartbeat notifications
        notif_ctx = self._pending_notification_context
        self._pending_notification_context = ""  # Consume once
        if notif_ctx:
            sections.append(
                f"Pending notifications to deliver (incorporate naturally):\n{notif_ctx}"
            )

        return "\n\n".join(sections)

    @staticmethod
    def _get_datetime_str() -> str:
        """Generate Indonesian datetime string for the current moment."""
        now = datetime.datetime.now()
        return (
            f"Sekarang: {now.strftime('%H:%M')}, "
            f"{_HARI[now.weekday()]}, "
            f"{now.day} {_BULAN[now.month]} {now.year}"
        )

    def update_user_profile(self, info: str) -> None:
        """Append information to USER.md.

        Args:
            info: Text to append to the user profile.
        """
        path = self._dir / "USER.md"
        try:
            current = ""
            if path.exists():
                current = path.read_text(encoding="utf-8").rstrip()
            updated = f"{current}\n{info.strip()}\n" if current else f"{info.strip()}\n"
            path.write_text(updated, encoding="utf-8")
            # Invalidate cache
            self._cache.pop("USER.md", None)
            logger.info("Updated USER.md with: %s", info[:80])
        except OSError as e:
            logger.error("Failed to update USER.md: %s", e)

    @property
    def prompts_dir(self) -> Path:
        """Return the prompts directory path."""
        return self._dir

    def set_memory_context(self, context: str) -> None:
        """Set memory context to be included in the next build() call.

        The context is consumed (cleared) after the next build() call.

        Args:
            context: Formatted memory context string.
        """
        self._pending_memory_context = context

    def set_notification_context(self, context: str) -> None:
        """Set notification context to be included in the next build() call.

        The context is consumed (cleared) after the next build() call.

        Args:
            context: Formatted notification context string.
        """
        self._pending_notification_context = context


def get_prompt_assembler() -> PromptAssembler:
    """Get the singleton PromptAssembler instance.

    Returns:
        The shared PromptAssembler.
    """
    global _instance
    if _instance is None:
        _instance = PromptAssembler()
    return _instance


def reset_prompt_assembler() -> None:
    """Reset the singleton (for testing)."""
    global _instance
    _instance = None
