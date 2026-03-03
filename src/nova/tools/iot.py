"""IoT device control tool for NOVA.

Dispatches voice commands to the appropriate driver:
- AC → TuyaCloudDriver (IR)
- TV Atas → TuyaCloudDriver (IR) + LGWebOSDriver (smart features)
- TV Bawah → LGWebOSDriver only (no IR hub)
"""

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env so TUYA_* and LG_TV_* vars are available via os.environ.
# Pydantic only loads NOVA_-prefixed vars; IoT vars use their own prefix.
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Lazy-initialized driver singletons
_tuya_driver = None
_tv_atas_webos = None
_tv_bawah_webos = None


def _get_tuya_driver():
    """Get or create the TuyaCloudDriver singleton."""
    global _tuya_driver
    if _tuya_driver is None:
        from nova.iot.tuya_cloud import TuyaCloudDriver
        _tuya_driver = TuyaCloudDriver()
    return _tuya_driver


def _get_tv_atas_webos():
    """Get or create the LG WebOS driver for TV Atas."""
    global _tv_atas_webos
    if _tv_atas_webos is None:
        from nova.iot.lg_webos import LGWebOSDriver
        ip = os.environ.get("LG_TV_ATAS_IP", "")
        if not ip:
            return None
        client_key = os.environ.get("LG_TV_ATAS_CLIENT_KEY", "") or None
        _tv_atas_webos = LGWebOSDriver(ip=ip, name="TV Atas", client_key=client_key)
    return _tv_atas_webos


def _get_tv_bawah_webos():
    """Get or create the LG WebOS driver for TV Bawah."""
    global _tv_bawah_webos
    if _tv_bawah_webos is None:
        from nova.iot.lg_webos import LGWebOSDriver
        ip = os.environ.get("LG_TV_BAWAH_IP", "")
        if not ip:
            return None
        client_key = os.environ.get("LG_TV_BAWAH_CLIENT_KEY", "") or None
        _tv_bawah_webos = LGWebOSDriver(
            ip=ip, name="TV Bawah", client_key=client_key,
        )
    return _tv_bawah_webos


# IR command mapping for TV navigation/control
_TV_IR_COMMANDS = {
    "volume_up": "Volume+",
    "volume_down": "Volume-",
    "channel_up": "Channel+",
    "channel_down": "Channel-",
    "home": "Home",
    "back": "Back",
    "menu": "Menu",
    "up": "Up",
    "down": "Down",
    "left": "Left",
    "right": "Right",
    "ok": "OK",
}

# AC mode names for user-friendly responses
_AC_MODE_NAMES = {
    "0": "dingin", "1": "panas", "2": "auto", "3": "kipas", "4": "kering",
    "cool": "0", "heat": "1", "auto": "2", "fan": "3", "dry": "4",
    "dingin": "0", "panas": "1", "kipas": "3", "kering": "4",
}

# AC fan speed names
_AC_FAN_NAMES = {
    "0": "auto", "1": "pelan", "2": "sedang", "3": "kencang",
    "auto": "0", "low": "1", "medium": "2", "high": "3",
    "pelan": "1", "sedang": "2", "kencang": "3",
}


async def control_device(
    device: str,
    action: str,
    value: str = "",
) -> str:
    """Control a smart home device.

    Args:
        device: Device name — "ac", "tv_atas", "tv_bawah".
        action: What to do — "on", "off", "set_temp", "set_mode", "set_fan",
                "volume_up", "volume_down", "set_volume", "channel_up",
                "channel_down", "open_app", "home", "back", "menu",
                "up", "down", "left", "right", "ok".
        value: Optional value — temperature (16-30), volume level (0-100),
               app name, mode number, fan speed number, etc.

    Returns:
        Result message for the LLM to relay to user.
    """
    device = device.strip().lower()
    action = action.strip().lower()
    value = value.strip()

    logger.info("IoT control_device: device=%s action=%s value=%s", device, action, value)

    if device == "ac":
        return await _handle_ac(action, value)
    elif device == "tv_atas":
        return await _handle_tv_atas(action, value)
    elif device == "tv_bawah":
        return await _handle_tv_bawah(action, value)
    else:
        return (
            f"Perangkat '{device}' tidak dikenali. "
            "Yang tersedia: ac, tv_atas, tv_bawah."
        )


async def _handle_ac(action: str, value: str) -> str:
    """Handle AC commands."""
    try:
        tuya = _get_tuya_driver()
    except Exception as e:
        return f"Gagal menghubungkan ke Tuya Cloud: {e}"

    try:
        if action == "on":
            return await tuya.send_ac_command(power=True)
        elif action == "off":
            return await tuya.send_ac_command(power=False)
        elif action == "set_temp":
            temp = _parse_int(value, 16, 30)
            if temp is None:
                return "Suhu AC harus antara 16-30 derajat."
            return await tuya.send_ac_command(temp=temp)
        elif action == "set_mode":
            mode = _resolve_ac_mode(value)
            if mode is None:
                return "Mode AC tidak valid. Pilihan: 0=dingin, 1=panas, 2=auto, 3=kipas, 4=kering."
            return await tuya.send_ac_command(mode=mode)
        elif action == "set_fan":
            fan = _resolve_ac_fan(value)
            if fan is None:
                return "Kecepatan kipas tidak valid. Pilihan: 0=auto, 1=pelan, 2=sedang, 3=kencang."
            return await tuya.send_ac_command(fan=fan)
        else:
            return (
                f"Aksi AC '{action}' tidak dikenali. "
                "Gunakan: on, off, set_temp, set_mode, set_fan."
            )
    except Exception as e:
        logger.error("AC command error: %s", e)
        return f"Gagal mengontrol AC: {e}"


async def _handle_tv_atas(action: str, value: str) -> str:
    """Handle TV Atas commands — uses IR + WebOS."""
    try:
        if action == "on":
            return await _tv_atas_ir("Power")
        elif action == "off":
            return await _tv_atas_webos_cmd("power_off")
        elif action == "open_app":
            return await _tv_atas_webos_app(value)
        elif action == "set_volume":
            vol = _parse_int(value, 0, 100)
            if vol is None:
                return "Volume harus antara 0-100."
            return await _tv_atas_webos_volume(vol)
        elif action == "volume_up":
            return await _tv_atas_webos_volume_step("up")
        elif action == "volume_down":
            return await _tv_atas_webos_volume_step("down")
        elif action in ("channel_up", "channel_down"):
            ir_cmd = _TV_IR_COMMANDS.get(action, action)
            return await _tv_atas_ir(ir_cmd)
        elif action in ("home", "back", "menu", "up", "down", "left", "right", "ok"):
            ir_cmd = _TV_IR_COMMANDS.get(action, action)
            return await _tv_atas_ir(ir_cmd)
        else:
            return f"Aksi TV '{action}' tidak dikenali."
    except Exception as e:
        logger.error("TV Atas command error: %s", e)
        return f"Gagal mengontrol TV Atas: {e}"


async def _handle_tv_bawah(action: str, value: str) -> str:
    """Handle TV Bawah commands — WebOS ONLY, no IR."""
    # TV Bawah cannot be powered on remotely
    if action == "on":
        return "TV Bawah harus dinyalakan manual dulu, Tuan. Setelah menyala, saya bisa kontrol."

    # No IR = no channel or navigation
    if action in ("channel_up", "channel_down", "up", "down", "left", "right", "ok", "menu"):
        return f"Aksi '{action}' tidak tersedia untuk TV Bawah (tidak ada remote IR)."

    webos = _get_tv_bawah_webos()
    if webos is None:
        return "IP TV Bawah belum dikonfigurasi (LG_TV_BAWAH_IP)."

    try:
        if action == "off":
            return await webos.power_off()
        elif action == "open_app":
            return await webos.launch_app(value)
        elif action == "set_volume":
            vol = _parse_int(value, 0, 100)
            if vol is None:
                return "Volume harus antara 0-100."
            return await webos.set_volume(vol)
        elif action == "volume_up":
            return await webos.volume_up()
        elif action == "volume_down":
            return await webos.volume_down()
        elif action == "home":
            # WebOS can handle home via API
            return await webos.launch_app("tv")
        elif action == "back":
            return "Aksi 'back' tidak tersedia untuk TV Bawah (tidak ada remote IR)."
        else:
            return f"Aksi TV '{action}' tidak dikenali."
    except ConnectionError as e:
        return f"Tidak bisa terhubung ke TV Bawah. Pastikan TV sudah menyala. ({e})"
    except Exception as e:
        logger.error("TV Bawah command error: %s", e)
        return f"Gagal mengontrol TV Bawah: {e}"


# ── TV Atas helpers ──────────────────────────────────────────────────


async def _tv_atas_ir(command: str) -> str:
    """Send IR command to TV Atas via Tuya."""
    try:
        tuya = _get_tuya_driver()
        return await tuya.send_tv_ir_command(command)
    except Exception as e:
        logger.error("TV Atas IR error: %s", e)
        return f"Gagal mengirim perintah IR ke TV Atas: {e}"


async def _tv_atas_webos_cmd(command: str) -> str:
    """Send WebOS command to TV Atas, fall back to IR if WebOS fails."""
    webos = _get_tv_atas_webos()
    if webos is not None:
        try:
            if command == "power_off":
                return await webos.power_off()
        except ConnectionError:
            logger.warning("TV Atas WebOS unreachable, falling back to IR.")

    # Fallback to IR
    if command == "power_off":
        return await _tv_atas_ir("Power")
    return f"Perintah '{command}' gagal."


async def _tv_atas_webos_app(app_name: str) -> str:
    """Launch app on TV Atas via WebOS."""
    webos = _get_tv_atas_webos()
    if webos is None:
        return "IP TV Atas belum dikonfigurasi (LG_TV_ATAS_IP)."
    try:
        return await webos.launch_app(app_name)
    except ConnectionError:
        return "TV Atas tidak merespons. Pastikan TV sudah menyala."


async def _tv_atas_webos_volume(level: int) -> str:
    """Set volume on TV Atas via WebOS, fall back to IR."""
    webos = _get_tv_atas_webos()
    if webos is not None:
        try:
            return await webos.set_volume(level)
        except ConnectionError:
            logger.warning("TV Atas WebOS unreachable for volume, falling back to IR.")

    # No precise volume via IR, just report the issue
    return "Tidak bisa set volume TV Atas. TV mungkin belum menyala."


async def _tv_atas_webos_volume_step(direction: str) -> str:
    """Volume up/down on TV Atas via WebOS, fall back to IR."""
    webos = _get_tv_atas_webos()
    if webos is not None:
        try:
            if direction == "up":
                return await webos.volume_up()
            else:
                return await webos.volume_down()
        except ConnectionError:
            logger.warning("TV Atas WebOS unreachable, volume via IR.")

    # Fallback to IR
    ir_cmd = "Volume+" if direction == "up" else "Volume-"
    return await _tv_atas_ir(ir_cmd)


# ── Value parsing helpers ────────────────────────────────────────────


def _parse_int(value: str, min_val: int, max_val: int) -> int | None:
    """Parse string to int within range, or return None."""
    try:
        n = int(value)
        if min_val <= n <= max_val:
            return n
        return None
    except (ValueError, TypeError):
        return None


def _resolve_ac_mode(value: str) -> int | None:
    """Resolve AC mode from number or name string."""
    v = value.lower().strip()
    # Direct number
    if v in ("0", "1", "2", "3", "4"):
        return int(v)
    # Named mode
    mapped = _AC_MODE_NAMES.get(v)
    if mapped is not None:
        return int(mapped)
    return None


def _resolve_ac_fan(value: str) -> int | None:
    """Resolve AC fan speed from number or name string."""
    v = value.lower().strip()
    if v in ("0", "1", "2", "3"):
        return int(v)
    mapped = _AC_FAN_NAMES.get(v)
    if mapped is not None:
        return int(mapped)
    return None
