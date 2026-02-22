"""Audio playback utility — plays MP3/WAV bytes through speakers."""

import asyncio
import logging
import os
import shutil
import sys
import tempfile

logger = logging.getLogger(__name__)


def _find_player() -> tuple[str, list[str]]:
    """Find an available audio player on the system.

    Returns:
        Tuple of (player_name, command_args) where command_args expects
        the file path appended.

    Raises:
        RuntimeError: If no supported audio player is found.
    """
    # mpv — primary player (target platform: Ubuntu)
    if shutil.which("mpv"):
        return "mpv", ["mpv", "--no-video", "--really-quiet"]

    # ffplay — common on Windows with ffmpeg installed
    if shutil.which("ffplay"):
        return "ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet"]

    # aplay — Linux fallback (WAV only, but works for basic playback)
    if shutil.which("aplay"):
        return "aplay", ["aplay", "-q"]

    # Windows — use the default media player via start command
    if sys.platform == "win32":
        return "start", ["cmd", "/c", "start", "", "/wait"]

    raise RuntimeError(
        "No audio player found. Install mpv (recommended): "
        "sudo apt install mpv (Linux) or winget install mpv (Windows)"
    )


async def play_audio(audio_bytes: bytes) -> None:
    """Play audio bytes through the system speakers.

    Saves to a temp file, plays via the best available player, then cleans up.

    Args:
        audio_bytes: MP3 or WAV audio data.
    """
    if not audio_bytes:
        logger.warning("play_audio called with empty audio bytes")
        return

    player_name, cmd = _find_player()
    logger.debug("Using audio player: %s", player_name)

    # Write to temp file
    suffix = ".mp3"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        tmp.write(audio_bytes)
        tmp.close()

        full_cmd = cmd + [tmp.name]
        logger.debug("Running: %s", " ".join(full_cmd))

        process = await asyncio.create_subprocess_exec(
            *full_cmd,
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await process.wait()

        if process.returncode != 0:
            logger.warning("Audio player %s exited with code %d", player_name, process.returncode)
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp.name)
        except OSError:
            pass
