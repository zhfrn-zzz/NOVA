"""System control tools — Windows-compatible commands for controlling the laptop.

All functions are async and return a human-readable status string.
"""

import asyncio
import logging
import subprocess
import sys
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────

_WIN = sys.platform == "win32"


async def _send_key(vk_code: int) -> None:
    """Send a virtual key press via PowerShell WScript.Shell."""
    cmd = (
        'powershell -Command "'
        "$shell = New-Object -ComObject WScript.Shell; "
        f'$shell.SendKeys([char]{vk_code})"'
    )
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )


def _popen(args: list[str]) -> None:
    """Fire-and-forget a subprocess."""
    subprocess.Popen(
        args,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── Volume ───────────────────────────────────────────────────────────

async def volume_up() -> str:
    """Increase system volume by ~10%.

    Returns:
        Status message.
    """
    if not _WIN:
        return "Volume control is only supported on Windows."
    await _send_key(175)  # VK_VOLUME_UP
    await _send_key(175)
    logger.info("Tool volume_up executed")
    return "Volume telah dinaikkan."


async def volume_down() -> str:
    """Decrease system volume by ~10%.

    Returns:
        Status message.
    """
    if not _WIN:
        return "Volume control is only supported on Windows."
    await _send_key(174)  # VK_VOLUME_DOWN
    await _send_key(174)
    logger.info("Tool volume_down executed")
    return "Volume telah diturunkan."


async def mute_unmute() -> str:
    """Toggle system audio mute.

    Returns:
        Status message.
    """
    if not _WIN:
        return "Mute control is only supported on Windows."
    await _send_key(173)  # VK_VOLUME_MUTE
    logger.info("Tool mute_unmute executed")
    return "Mute telah di-toggle."


# ── Media Controls ───────────────────────────────────────────────────

async def play_pause_media() -> str:
    """Toggle media play/pause.

    Returns:
        Status message.
    """
    if not _WIN:
        return "Media control is only supported on Windows."
    await _send_key(179)  # VK_MEDIA_PLAY_PAUSE
    logger.info("Tool play_pause_media executed")
    return "Media play/pause di-toggle."


async def next_track() -> str:
    """Skip to next media track.

    Returns:
        Status message.
    """
    if not _WIN:
        return "Media control is only supported on Windows."
    await _send_key(176)  # VK_MEDIA_NEXT_TRACK
    logger.info("Tool next_track executed")
    return "Berpindah ke track selanjutnya."


async def previous_track() -> str:
    """Go to previous media track.

    Returns:
        Status message.
    """
    if not _WIN:
        return "Media control is only supported on Windows."
    await _send_key(177)  # VK_MEDIA_PREV_TRACK
    logger.info("Tool previous_track executed")
    return "Berpindah ke track sebelumnya."


# ── Applications ─────────────────────────────────────────────────────

# Map of friendly app names to executables / start commands
_APP_MAP: dict[str, list[str]] = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "spotify": ["cmd", "/c", "start", "spotify:"],
    "discord": ["cmd", "/c", "start", "discord:"],
    "whatsapp": ["cmd", "/c", "start", "whatsapp:"],
    "vscode": ["cmd", "/c", "code"],
    "explorer": ["explorer.exe"],
    "paint": ["mspaint.exe"],
    "settings": ["cmd", "/c", "start", "ms-settings:"],
    "task manager": ["taskmgr.exe"],
}


async def open_app(app_name: str) -> str:
    """Open an application by its common name.

    Args:
        app_name: Friendly app name (e.g. 'spotify', 'notepad', 'vscode').

    Returns:
        Status message.
    """
    key = app_name.strip().lower()
    cmd_args = _APP_MAP.get(key)
    if cmd_args is None:
        # Fallback: try to run it directly
        cmd_args = ["cmd", "/c", "start", key]

    try:
        _popen(cmd_args)
        logger.info("Tool open_app executed: %s", app_name)
        return f"{app_name} telah dibuka."
    except Exception as e:
        logger.error("Failed to open app %s: %s", app_name, e)
        return f"Gagal membuka {app_name}: {e}"


async def open_browser() -> str:
    """Open the default web browser.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen(["cmd", "/c", "start", "https://www.google.com"])
        else:
            _popen(["xdg-open", "https://www.google.com"])
        logger.info("Tool open_browser executed")
        return "Browser telah dibuka."
    except Exception as e:
        logger.error("Failed to open browser: %s", e)
        return f"Gagal membuka browser: {e}"


async def open_url(url: str) -> str:
    """Open a URL in the default web browser.

    Args:
        url: The URL to open.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen(["cmd", "/c", "start", url])
        else:
            _popen(["xdg-open", url])
        logger.info("Tool open_url executed: %s", url)
        return f"Membuka {url}."
    except Exception as e:
        logger.error("Failed to open URL %s: %s", url, e)
        return f"Gagal membuka URL: {e}"


async def open_terminal() -> str:
    """Open a terminal window.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen(["wt"])
        else:
            _popen(["gnome-terminal"])
        logger.info("Tool open_terminal executed")
        return "Terminal telah dibuka."
    except Exception as e:
        logger.error("Failed to open terminal: %s", e)
        return f"Gagal membuka terminal: {e}"


async def open_file_manager() -> str:
    """Open the file manager (Explorer on Windows).

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen(["explorer.exe"])
        else:
            _popen(["nautilus"])
        logger.info("Tool open_file_manager executed")
        return "File manager telah dibuka."
    except Exception as e:
        logger.error("Failed to open file manager: %s", e)
        return f"Gagal membuka file manager: {e}"


# ── System Power ─────────────────────────────────────────────────────

async def lock_screen() -> str:
    """Lock the workstation screen.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen(["rundll32.exe", "user32.dll,LockWorkStation"])
        else:
            _popen(["loginctl", "lock-session"])
        logger.info("Tool lock_screen executed")
        return "Layar telah dikunci."
    except Exception as e:
        logger.error("Failed to lock screen: %s", e)
        return f"Gagal mengunci layar: {e}"


async def shutdown_pc(delay_seconds: int = 60) -> str:
    """Schedule a system shutdown with a countdown.

    Args:
        delay_seconds: Seconds before shutdown (default 60). Use 0 for immediate.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen(["shutdown", "/s", "/t", str(delay_seconds)])
        else:
            _popen(["shutdown", "-h", f"+{max(delay_seconds // 60, 1)}"])
        logger.info("Tool shutdown_pc executed: delay=%ds", delay_seconds)
        return f"Komputer akan dimatikan dalam {delay_seconds} detik."
    except Exception as e:
        logger.error("Failed to schedule shutdown: %s", e)
        return f"Gagal menjadwalkan shutdown: {e}"


async def restart_pc(delay_seconds: int = 60) -> str:
    """Schedule a system restart with a countdown.

    Args:
        delay_seconds: Seconds before restart (default 60). Use 0 for immediate.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen(["shutdown", "/r", "/t", str(delay_seconds)])
        else:
            _popen(["shutdown", "-r", f"+{max(delay_seconds // 60, 1)}"])
        logger.info("Tool restart_pc executed: delay=%ds", delay_seconds)
        return f"Komputer akan di-restart dalam {delay_seconds} detik."
    except Exception as e:
        logger.error("Failed to schedule restart: %s", e)
        return f"Gagal menjadwalkan restart: {e}"


async def sleep_pc() -> str:
    """Put the PC to sleep.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            _popen([
                "powershell", "-Command",
                "Add-Type -AssemblyName System.Windows.Forms; "
                "[System.Windows.Forms.Application]::SetSuspendState("
                "'Suspend', $false, $false)",
            ])
        else:
            _popen(["systemctl", "suspend"])
        logger.info("Tool sleep_pc executed")
        return "Komputer akan masuk mode sleep."
    except Exception as e:
        logger.error("Failed to put PC to sleep: %s", e)
        return f"Gagal sleep: {e}"


# ── Screenshot ───────────────────────────────────────────────────────

async def take_screenshot() -> str:
    """Take a screenshot and save to ~/Pictures/Screenshots/.

    Returns:
        Status message with the file path.
    """
    screenshots_dir = Path.home() / "Pictures" / "Screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = screenshots_dir / f"screenshot_{timestamp}.png"

    try:
        if _WIN:
            # Use PowerShell to capture the screen
            ps_cmd = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
                "$bitmap = New-Object System.Drawing.Bitmap($screen.Width, $screen.Height); "
                "$graphics = [System.Drawing.Graphics]::FromImage($bitmap); "
                "$graphics.CopyFromScreen($screen.Location, "
                "[System.Drawing.Point]::Empty, $screen.Size); "
                f"$bitmap.Save('{filepath}'); "
                "$graphics.Dispose(); $bitmap.Dispose()"
            )
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-Command", ps_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
            if proc.returncode != 0:
                return f"Screenshot gagal: {stderr.decode().strip()}"
        else:
            proc = await asyncio.create_subprocess_exec(
                "gnome-screenshot", "-f", str(filepath),
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()

        logger.info("Tool take_screenshot executed: %s", filepath)
        return f"Screenshot tersimpan di {filepath}."
    except Exception as e:
        logger.error("Failed to take screenshot: %s", e)
        return f"Gagal mengambil screenshot: {e}"


# ── Timer ────────────────────────────────────────────────────────────

async def set_timer(seconds: int, label: str = "Timer") -> str:
    """Set a countdown timer that shows a notification when done.

    Runs in the background — returns immediately with confirmation.

    Args:
        seconds: Duration in seconds.
        label: Description for the timer notification.

    Returns:
        Status message.
    """
    if seconds <= 0:
        return "Durasi timer harus lebih dari 0 detik."

    async def _timer_task() -> None:
        await asyncio.sleep(seconds)
        logger.info("Timer finished: %s (%ds)", label, seconds)
        try:
            if _WIN:
                # Use PowerShell toast notification
                ps_cmd = (
                    "Add-Type -AssemblyName System.Windows.Forms; "
                    "$notify = New-Object System.Windows.Forms.NotifyIcon; "
                    "$notify.Icon = [System.Drawing.SystemIcons]::Information; "
                    "$notify.Visible = $true; "
                    f"$notify.ShowBalloonTip(5000, 'NOVA Timer', '{label}', "
                    "'Info'); Start-Sleep -Seconds 6; $notify.Dispose()"
                )
                proc = await asyncio.create_subprocess_exec(
                    "powershell", "-Command", ps_cmd,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
            else:
                proc = await asyncio.create_subprocess_exec(
                    "notify-send", "NOVA Timer", label,
                    stdout=asyncio.subprocess.DEVNULL,
                    stderr=asyncio.subprocess.DEVNULL,
                )
                await proc.wait()
        except Exception:
            logger.debug("Timer notification failed", exc_info=True)

    asyncio.create_task(_timer_task())
    logger.info("Tool set_timer executed: %ds, label=%r", seconds, label)

    if seconds >= 60:
        mins = seconds // 60
        secs = seconds % 60
        time_str = f"{mins} menit" + (f" {secs} detik" if secs else "")
    else:
        time_str = f"{seconds} detik"

    return f"Timer {label} telah diset untuk {time_str}."


# ── Test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time

    async def main() -> None:
        print("=== System Control Tools Test ===")
        print("Testing volume up...")
        print(await volume_up())
        time.sleep(1)
        print("Testing volume down...")
        print(await volume_down())
        time.sleep(1)
        print("Testing mute toggle...")
        print(await mute_unmute())
        time.sleep(1)
        print("Testing open browser...")
        print(await open_browser())

    asyncio.run(main())
