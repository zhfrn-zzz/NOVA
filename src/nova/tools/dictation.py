"""Dictation tool — types text into the active window using pyautogui.

The LLM receives transcribed speech, cleans it, and calls dictate() to
simulate keyboard input into whatever app is currently focused.
"""

import asyncio
import logging

logger = logging.getLogger(__name__)


async def dictate(text: str) -> str:
    """Type the given text into the currently active window.

    Uses pyautogui.write() for ASCII and pyperclip+hotkey for Unicode text.

    Args:
        text: The text to type.

    Returns:
        Confirmation message.
    """
    if not text.strip():
        return "Tidak ada teks untuk diketik."

    try:
        import pyautogui

        # Small delay to let user focus the target window
        await asyncio.sleep(0.5)

        # pyautogui.write() only handles ASCII; for Unicode, use clipboard
        try:
            pyautogui.write(text, interval=0.02)
        except Exception:
            # Fallback: copy to clipboard and paste
            import pyperclip

            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")

        logger.info("Tool dictate → %d chars", len(text))
        return f"Teks berhasil diketik: {text[:50]}{'...' if len(text) > 50 else ''}"
    except ImportError:
        return "pyautogui belum terinstall. Jalankan: pip install pyautogui"
    except Exception as e:
        logger.error("Failed to dictate text: %s", e)
        return f"Gagal mengetik teks: {e}"


if __name__ == "__main__":

    async def main() -> None:
        print("=== Dictation Tool Test ===")
        print("Typing in 3 seconds... switch to a text editor!")
        await asyncio.sleep(3)
        print(await dictate("Hello World from NOVA!"))

    asyncio.run(main())
