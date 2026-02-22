"""System info tools — battery, RAM, storage, IP, uptime via psutil.

All functions are async and return human-readable status strings.
"""

import asyncio
import logging
import socket
import sys
from datetime import datetime

import psutil

logger = logging.getLogger(__name__)


async def get_battery_level() -> str:
    """Get battery percentage and charging status.

    Returns:
        Battery info string, e.g. "Baterai: 75%, sedang mengisi."
    """
    try:
        battery = psutil.sensors_battery()
        if battery is None:
            return "Tidak ada baterai terdeteksi (kemungkinan PC desktop)."
        pct = round(battery.percent)
        if battery.power_plugged:
            charging = "sedang mengisi (charging)"
        else:
            charging = "tidak mengisi (discharging)"
        logger.info("Tool get_battery_level → %d%%, plugged=%s", pct, battery.power_plugged)
        return f"Baterai: {pct}%, {charging}."
    except Exception as e:
        logger.error("Failed to get battery info: %s", e)
        return f"Gagal mendapatkan info baterai: {e}"


async def get_ram_usage() -> str:
    """Get current RAM usage.

    Returns:
        RAM usage string, e.g. "RAM: 2.1 GB / 4.0 GB (53%)."
    """
    try:
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        pct = mem.percent
        logger.info("Tool get_ram_usage → %.1f/%.1f GB (%s%%)", used_gb, total_gb, pct)
        return f"RAM: {used_gb:.1f} GB / {total_gb:.1f} GB ({pct}% terpakai)."
    except Exception as e:
        logger.error("Failed to get RAM info: %s", e)
        return f"Gagal mendapatkan info RAM: {e}"


async def get_storage_info() -> str:
    """Get disk usage for the primary drive.

    Returns:
        Storage info string with used/total/free in GB.
    """
    try:
        path = "C:\\" if sys.platform == "win32" else "/"
        disk = psutil.disk_usage(path)
        used_gb = disk.used / (1024 ** 3)
        total_gb = disk.total / (1024 ** 3)
        free_gb = disk.free / (1024 ** 3)
        pct = disk.percent
        logger.info("Tool get_storage_info → %.1f/%.1f GB", used_gb, total_gb)
        return (
            f"Storage: {used_gb:.1f} GB / {total_gb:.1f} GB terpakai "
            f"({free_gb:.1f} GB tersisa, {pct}%)."
        )
    except Exception as e:
        logger.error("Failed to get storage info: %s", e)
        return f"Gagal mendapatkan info storage: {e}"


async def get_ip_address() -> str:
    """Get local and public IP addresses.

    Returns:
        IP address info string.
    """
    try:
        # Local IP
        local_ip = socket.gethostbyname(socket.gethostname())
        if local_ip.startswith("127."):
            # Fallback: connect to external to find local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            finally:
                s.close()

        # Public IP via httpx (async)
        import httpx

        public_ip = "tidak tersedia"
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get("https://ifconfig.me/ip")
                if resp.status_code == 200:
                    public_ip = resp.text.strip()
        except Exception:
            logger.debug("Failed to fetch public IP", exc_info=True)

        logger.info("Tool get_ip_address → local=%s, public=%s", local_ip, public_ip)
        return f"IP lokal: {local_ip}, IP publik: {public_ip}."
    except Exception as e:
        logger.error("Failed to get IP address: %s", e)
        return f"Gagal mendapatkan IP address: {e}"


async def get_system_uptime() -> str:
    """Get system uptime since last boot.

    Returns:
        Uptime string, e.g. "Sistem sudah menyala selama 3 jam 25 menit."
    """
    try:
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        total_seconds = int(uptime.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60

        if hours > 0:
            result = f"Sistem sudah menyala selama {hours} jam {minutes} menit."
        else:
            result = f"Sistem sudah menyala selama {minutes} menit."

        logger.info("Tool get_system_uptime → %s", result)
        return result
    except Exception as e:
        logger.error("Failed to get uptime: %s", e)
        return f"Gagal mendapatkan uptime: {e}"


if __name__ == "__main__":

    async def main() -> None:
        print("=== System Info Tools Test ===")
        print(await get_battery_level())
        print(await get_ram_usage())
        print(await get_storage_info())
        print(await get_ip_address())
        print(await get_system_uptime())

    asyncio.run(main())
