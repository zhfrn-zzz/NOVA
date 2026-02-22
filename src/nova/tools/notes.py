"""Quick notes tool — append, read, and clear notes stored in ~/.nova/notes.txt.

All functions are async and return human-readable status strings.
"""

import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

_NOTES_DIR = Path.home() / ".nova"
_NOTES_FILE = _NOTES_DIR / "notes.txt"


async def add_note(text: str) -> str:
    """Append a note with a timestamp to the notes file.

    Args:
        text: The note content to save.

    Returns:
        Confirmation message.
    """
    try:
        _NOTES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        line = f"[{timestamp}] {text.strip()}\n"
        with open(_NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(line)
        logger.info("Tool add_note → %s", text.strip())
        return f"Catatan tersimpan: {text.strip()}"
    except Exception as e:
        logger.error("Failed to add note: %s", e)
        return f"Gagal menyimpan catatan: {e}"


async def get_notes() -> str:
    """Return the last 10 notes from the notes file.

    Returns:
        Formatted string of recent notes, or a message if none exist.
    """
    try:
        if not _NOTES_FILE.exists():
            return "Belum ada catatan."
        lines = _NOTES_FILE.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return "Belum ada catatan."
        last_10 = lines[-10:]
        logger.info("Tool get_notes → %d notes", len(last_10))
        return "Catatan terakhir:\n" + "\n".join(last_10)
    except Exception as e:
        logger.error("Failed to read notes: %s", e)
        return f"Gagal membaca catatan: {e}"


async def clear_notes() -> str:
    """Clear all saved notes.

    Returns:
        Confirmation message.
    """
    try:
        if _NOTES_FILE.exists():
            _NOTES_FILE.write_text("", encoding="utf-8")
        logger.info("Tool clear_notes executed")
        return "Semua catatan telah dihapus."
    except Exception as e:
        logger.error("Failed to clear notes: %s", e)
        return f"Gagal menghapus catatan: {e}"


if __name__ == "__main__":
    import asyncio

    async def main() -> None:
        print("=== Notes Tools Test ===")
        print(await add_note("Beli kopi besok"))
        print(await add_note("Meeting jam 3 sore"))
        print(await get_notes())
        print(await clear_notes())
        print(await get_notes())

    asyncio.run(main())
