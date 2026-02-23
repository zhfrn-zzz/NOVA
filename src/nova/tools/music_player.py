"""Music playback tool — search and play songs via YouTube Music.

Uses yt-dlp to resolve song queries to YouTube video IDs, then opens
YouTube Music in the default browser for auto-playback. Playback control
uses global media keys via pyautogui.
"""

import asyncio
import logging
import webbrowser

logger = logging.getLogger(__name__)

_YTDLP_TIMEOUT = 10.0  # seconds


async def play_music(query: str) -> str:
    """Search for a song and play it on YouTube Music.

    Uses yt-dlp to search YouTube for the query, extracts the video ID,
    and opens the corresponding YouTube Music URL in the default browser.

    Args:
        query: Song search query, e.g. "About You The 1975".

    Returns:
        Status message with the song URL or an error.
    """
    if not query.strip():
        return "Tidak ada lagu yang diminta."

    logger.info("Music play request: %r", query)

    try:
        # Run yt-dlp to search YouTube and get the video ID
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "yt-dlp",
                f"ytsearch:{query}",
                "--get-id",
                "--no-warnings",
                "--no-playlist",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=_YTDLP_TIMEOUT,
        )
        stdout, stderr = await proc.communicate()
        video_id = stdout.decode("utf-8", errors="replace").strip()

        if not video_id:
            err = stderr.decode("utf-8", errors="replace").strip()
            logger.warning("yt-dlp returned no video ID: %s", err)
            return f"Tidak menemukan lagu untuk: {query}"

        # Take only the first ID if multiple lines
        video_id = video_id.splitlines()[0].strip()

        url = f"https://music.youtube.com/watch?v={video_id}"
        logger.info("Opening YouTube Music: %s", url)
        webbrowser.open(url)

        return f"Memutar lagu: {query} — {url}"

    except TimeoutError:
        logger.warning("yt-dlp timed out after %.1fs", _YTDLP_TIMEOUT)
        return "Pencarian lagu terlalu lama, coba lagi."
    except FileNotFoundError:
        logger.error("yt-dlp not found in PATH")
        return "yt-dlp belum terinstall. Jalankan: pip install yt-dlp"
    except Exception as e:
        logger.error("Failed to play music: %s", e)
        return f"Gagal memutar lagu: {e}"


async def pause_resume_music() -> str:
    """Toggle play/pause on the currently playing music.

    Sends the global media 'playpause' key via pyautogui.

    Returns:
        Status message.
    """
    try:
        import pyautogui

        await asyncio.to_thread(pyautogui.press, "playpause")
        logger.info("Tool pause_resume_music executed")
        return "Musik di-pause/resume."
    except ImportError:
        return "pyautogui belum terinstall. Jalankan: pip install pyautogui"
    except Exception as e:
        logger.error("Failed to pause/resume music: %s", e)
        return f"Gagal pause/resume musik: {e}"


async def skip_track() -> str:
    """Skip to the next track.

    Sends the global media 'nexttrack' key via pyautogui.

    Returns:
        Status message.
    """
    try:
        import pyautogui

        await asyncio.to_thread(pyautogui.press, "nexttrack")
        logger.info("Tool skip_track executed")
        return "Beralih ke lagu selanjutnya."
    except ImportError:
        return "pyautogui belum terinstall. Jalankan: pip install pyautogui"
    except Exception as e:
        logger.error("Failed to skip track: %s", e)
        return f"Gagal skip lagu: {e}"


async def previous_music_track() -> str:
    """Go back to the previous track.

    Sends the global media 'prevtrack' key via pyautogui.

    Returns:
        Status message.
    """
    try:
        import pyautogui

        await asyncio.to_thread(pyautogui.press, "prevtrack")
        logger.info("Tool previous_music_track executed")
        return "Kembali ke lagu sebelumnya."
    except ImportError:
        return "pyautogui belum terinstall. Jalankan: pip install pyautogui"
    except Exception as e:
        logger.error("Failed to go to previous track: %s", e)
        return f"Gagal ke lagu sebelumnya: {e}"


async def stop_music() -> str:
    """Stop the currently playing music.

    Sends the global media 'stop' key via pyautogui.

    Returns:
        Status message.
    """
    try:
        import pyautogui

        await asyncio.to_thread(pyautogui.press, "stop")
        logger.info("Tool stop_music executed")
        return "Musik dihentikan."
    except ImportError:
        return "pyautogui belum terinstall. Jalankan: pip install pyautogui"
    except Exception as e:
        logger.error("Failed to stop music: %s", e)
        return f"Gagal menghentikan musik: {e}"


if __name__ == "__main__":

    async def main() -> None:
        print("=== Music Player Tools Test ===")
        print(await play_music("About You The 1975"))

    asyncio.run(main())
