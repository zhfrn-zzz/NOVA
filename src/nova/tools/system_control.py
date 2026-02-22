"""System control tools — Windows-compatible commands for controlling the laptop.

All functions are async and return a human-readable status string.
"""

import asyncio
import logging
import subprocess
import sys

logger = logging.getLogger(__name__)


async def volume_up() -> str:
    """Increase system volume by ~10%.

    Returns:
        Status message.
    """
    if sys.platform != "win32":
        return "Volume control is only supported on Windows."

    # Simulate pressing volume-up key twice (~10%) via PowerShell
    cmd = (
        'powershell -Command "'
        "$shell = New-Object -ComObject WScript.Shell; "
        "$shell.SendKeys([char]175); "  # VK_VOLUME_UP
        '$shell.SendKeys([char]175)"'
    )
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    logger.info("Tool volume_up executed")
    return "Volume telah dinaikkan."


async def volume_down() -> str:
    """Decrease system volume by ~10%.

    Returns:
        Status message.
    """
    if sys.platform != "win32":
        return "Volume control is only supported on Windows."

    cmd = (
        'powershell -Command "'
        "$shell = New-Object -ComObject WScript.Shell; "
        "$shell.SendKeys([char]174); "  # VK_VOLUME_DOWN
        '$shell.SendKeys([char]174)"'
    )
    await asyncio.create_subprocess_shell(
        cmd,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.DEVNULL,
    )
    logger.info("Tool volume_down executed")
    return "Volume telah diturunkan."


async def open_browser() -> str:
    """Open the default web browser.

    Returns:
        Status message.
    """
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                ["cmd", "/c", "start", "https://www.google.com"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["xdg-open", "https://www.google.com"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        logger.info("Tool open_browser executed")
        return "Browser telah dibuka."
    except Exception as e:
        logger.error("Failed to open browser: %s", e)
        return f"Gagal membuka browser: {e}"


async def open_terminal() -> str:
    """Open a terminal window.

    Returns:
        Status message.
    """
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                ["wt"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["gnome-terminal"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        logger.info("Tool open_terminal executed")
        return "Terminal telah dibuka."
    except Exception as e:
        logger.error("Failed to open terminal: %s", e)
        return f"Gagal membuka terminal: {e}"


async def lock_screen() -> str:
    """Lock the workstation screen.

    Returns:
        Status message.
    """
    try:
        if sys.platform == "win32":
            subprocess.Popen(
                ["rundll32.exe", "user32.dll,LockWorkStation"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        else:
            subprocess.Popen(
                ["loginctl", "lock-session"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        logger.info("Tool lock_screen executed")
        return "Layar telah dikunci."
    except Exception as e:
        logger.error("Failed to lock screen: %s", e)
        return f"Gagal mengunci layar: {e}"


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
        print("Testing open browser...")
        print(await open_browser())

    asyncio.run(main())
