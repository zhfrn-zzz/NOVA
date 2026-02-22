"""Network control tools — Wi-Fi on/off and status via netsh (Windows).

All functions are async and return human-readable status strings.
"""

import asyncio
import logging
import sys

logger = logging.getLogger(__name__)

_WIN = sys.platform == "win32"


async def _run_cmd(args: list[str]) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
    )


async def wifi_on() -> str:
    """Enable the Wi-Fi interface.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            rc, _, err = await _run_cmd([
                "netsh", "interface", "set", "interface", "Wi-Fi", "enable",
            ])
            if rc != 0:
                return f"Gagal mengaktifkan Wi-Fi: {err}"
            logger.info("Tool wifi_on executed")
            return "Wi-Fi telah diaktifkan."
        else:
            rc, _, err = await _run_cmd(["nmcli", "radio", "wifi", "on"])
            if rc != 0:
                return f"Gagal mengaktifkan Wi-Fi: {err}"
            logger.info("Tool wifi_on executed")
            return "Wi-Fi telah diaktifkan."
    except Exception as e:
        logger.error("Failed to enable Wi-Fi: %s", e)
        return f"Gagal mengaktifkan Wi-Fi: {e}"


async def wifi_off() -> str:
    """Disable the Wi-Fi interface.

    Returns:
        Status message.
    """
    try:
        if _WIN:
            rc, _, err = await _run_cmd([
                "netsh", "interface", "set", "interface", "Wi-Fi", "enable",
            ])
            # First ensure it's not already in a weird state, then disable
            rc, _, err = await _run_cmd([
                "netsh", "interface", "set", "interface", "Wi-Fi", "disable",
            ])
            if rc != 0:
                return f"Gagal menonaktifkan Wi-Fi: {err}"
            logger.info("Tool wifi_off executed")
            return "Wi-Fi telah dinonaktifkan."
        else:
            rc, _, err = await _run_cmd(["nmcli", "radio", "wifi", "off"])
            if rc != 0:
                return f"Gagal menonaktifkan Wi-Fi: {err}"
            logger.info("Tool wifi_off executed")
            return "Wi-Fi telah dinonaktifkan."
    except Exception as e:
        logger.error("Failed to disable Wi-Fi: %s", e)
        return f"Gagal menonaktifkan Wi-Fi: {e}"


async def get_wifi_status() -> str:
    """Get current Wi-Fi connection status and SSID.

    Returns:
        Wi-Fi status string with connected SSID or disconnected message.
    """
    try:
        if _WIN:
            rc, stdout, _ = await _run_cmd([
                "netsh", "wlan", "show", "interfaces",
            ])
            if rc != 0:
                return "Gagal mendapatkan status Wi-Fi."

            # Parse the output for SSID and State
            ssid = None
            state = None
            for line in stdout.splitlines():
                line = line.strip()
                if line.startswith("SSID") and "BSSID" not in line:
                    ssid = line.split(":", 1)[1].strip() if ":" in line else None
                if line.startswith("State"):
                    state = line.split(":", 1)[1].strip() if ":" in line else None

            if state and "connected" in state.lower():
                logger.info("Tool get_wifi_status → connected to %s", ssid)
                return f"Wi-Fi terhubung ke: {ssid}."
            else:
                logger.info("Tool get_wifi_status → disconnected")
                return "Wi-Fi tidak terhubung."
        else:
            rc, stdout, _ = await _run_cmd(["nmcli", "-t", "-f", "ACTIVE,SSID", "dev", "wifi"])
            for line in stdout.splitlines():
                if line.startswith("yes:"):
                    ssid = line.split(":", 1)[1]
                    logger.info("Tool get_wifi_status → connected to %s", ssid)
                    return f"Wi-Fi terhubung ke: {ssid}."
            logger.info("Tool get_wifi_status → disconnected")
            return "Wi-Fi tidak terhubung."
    except Exception as e:
        logger.error("Failed to get Wi-Fi status: %s", e)
        return f"Gagal mendapatkan status Wi-Fi: {e}"


if __name__ == "__main__":

    async def main() -> None:
        print("=== Network Control Tools Test ===")
        print(await get_wifi_status())

    asyncio.run(main())
