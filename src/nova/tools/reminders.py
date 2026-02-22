"""Reminders tool — schedule a reminder that speaks via TTS and shows a notification.

Uses asyncio for scheduling and plyer for cross-platform notifications.
"""

import asyncio
import logging
import sys

logger = logging.getLogger(__name__)


async def set_reminder(minutes: int, message: str) -> str:
    """Schedule a reminder that fires after the given number of minutes.

    When the timer fires, a system notification is shown and (if available)
    TTS speaks the reminder message.

    Args:
        minutes: Minutes from now until the reminder fires.
        message: The reminder message to speak/show.

    Returns:
        Confirmation message.
    """
    if minutes <= 0:
        return "Waktu reminder harus lebih dari 0 menit."

    async def _reminder_task() -> None:
        await asyncio.sleep(minutes * 60)
        logger.info("Reminder fired: %s", message)

        # Show system notification
        try:
            if sys.platform == "win32":
                # Use PowerShell toast notification (no extra deps on Windows)
                ps_cmd = (
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$notify = New-Object System.Windows.Forms.NotifyIcon; "
                    "$notify.Icon = [System.Drawing.SystemIcons]::Information; "
                    "$notify.Visible = $true; "
                    f"$notify.ShowBalloonTip(5000, 'NOVA Reminder', "
                    f"'{message.replace(chr(39), '')}', 'Info'); "
                    "Start-Sleep -Seconds 6; $notify.Dispose()"
                )
                proc = await asyncio.create_subprocess_exec(
                    "powershell", "-Command", ps_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            else:
                proc = await asyncio.create_subprocess_exec(
                    "notify-send", "NOVA Reminder", message,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
        except Exception:
            logger.debug("Notification failed", exc_info=True)

        # Try TTS playback of the reminder
        try:
            from nova.providers.tts.edge_tts_provider import EdgeTTSProvider

            tts = EdgeTTSProvider()
            audio_data = await tts.synthesize(f"Reminder: {message}")
            if audio_data:
                from nova.audio.playback import play_audio
                await play_audio(audio_data)
        except Exception:
            logger.debug("TTS reminder playback failed", exc_info=True)

    asyncio.create_task(_reminder_task())
    logger.info("Tool set_reminder → %d min, msg=%r", minutes, message)
    return f"Reminder diset untuk {minutes} menit lagi: {message}"


if __name__ == "__main__":

    async def main() -> None:
        print("=== Reminders Tool Test ===")
        print(await set_reminder(1, "Istirahat sebentar"))
        print("Waiting for reminder...")
        await asyncio.sleep(65)

    asyncio.run(main())
