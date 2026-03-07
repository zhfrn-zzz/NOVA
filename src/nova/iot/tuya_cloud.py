"""Tuya Cloud IR driver — sends commands to IR sub-devices via the Tuya Cloud API.

Used for AC control (power, temperature, mode, fan) and TV IR commands
(power, volume, channel, navigation). IR sub-devices CANNOT be controlled
locally — they require the cloud API.
"""

import asyncio
import logging
import os
from pathlib import Path

import tinytuya

logger = logging.getLogger(__name__)

# Device IDs
_IR_HUB_ID = "a37d9677a6a3498269xabd"
_AC_REMOTE_ID = "a387dddea7a6953cf4yslm"
_TV_REMOTE_ID = "a31c92a675b6ccc8504k2v"

# AC mode mapping
AC_MODES = {0: "cool", 1: "heat", 2: "auto", 3: "fan", 4: "dry"}
AC_FAN_SPEEDS = {0: "auto", 1: "low", 2: "medium", 3: "high"}

# Tuya IR category ID for TV remotes
_TV_CATEGORY_ID = 2

# Map high-level command names → Tuya raw key names
_TV_KEY_MAP = {
    "Power": "power",
    "Volume+": "volume+",
    "Volume-": "volume-",
    "Channel+": "channel+",
    "Channel-": "channel-",
    "Up": "up",
    "Down": "down",
    "Left": "left",
    "Right": "right",
    "OK": "ok",
    "Home": "home",
    "Back": "back",
    "Menu": "menu",
    "Mute": "mute",
}


class TuyaCloudDriver:
    """Controls Tuya IR sub-devices via Cloud API.

    Uses tinytuya.Cloud for authentication, then calls the Tuya IR-specific
    endpoints directly via _tuyaplatform().
    """

    def __init__(
        self,
        access_id: str | None = None,
        access_key: str | None = None,
        region: str | None = None,
    ) -> None:
        self._access_id = access_id or os.environ.get("TUYA_ACCESS_ID", "")
        self._access_key = access_key or os.environ.get("TUYA_ACCESS_KEY", "")
        self._region = region or os.environ.get("TUYA_REGION", "eu")
        self._cloud: tinytuya.Cloud | None = None
        # Tracked AC state for combined IR signals (scenes/command endpoint).
        # Each IR blast carries the full state, so we cache last-known values.
        self._ac_state: dict[str, str] = {
            "power": "1",
            "mode": "0",   # cool
            "temp": "24",
            "wind": "0",   # auto
        }

    def _get_cloud(self) -> tinytuya.Cloud:
        """Lazy-init the tinytuya Cloud client."""
        if self._cloud is None:
            from dotenv import load_dotenv
            load_dotenv(Path(__file__).resolve().parents[3] / ".env")

            # Re-read env vars after load_dotenv (in case they weren't set before)
            if not self._access_id:
                self._access_id = os.environ.get("TUYA_ACCESS_ID", "")
            if not self._access_key:
                self._access_key = os.environ.get("TUYA_ACCESS_KEY", "")

            if not self._access_id or not self._access_key:
                raise RuntimeError("TUYA_ACCESS_ID dan TUYA_ACCESS_KEY belum diset.")
            self._cloud = tinytuya.Cloud(
                apiRegion=self._region,
                apiKey=self._access_id,
                apiSecret=self._access_key,
            )
            logger.info("Tuya Cloud client initialized (region=%s)", self._region)
        return self._cloud

    def _post(self, uri: str, post: dict) -> dict:
        """POST to Tuya Cloud v2.0 API."""
        cloud = self._get_cloud()
        return cloud._tuyaplatform(uri, action="POST", post=post, ver="v2.0")

    def _send_ir_command_sync(
        self, ir_id: str, remote_id: str, code: str, value: str | int = "",
    ) -> dict:
        """Send an IR command via Tuya Cloud API (synchronous).

        Uses the standard IR remote command endpoint.
        """
        uri = f"infrareds/{ir_id}/remotes/{remote_id}/command"
        body = {"code": code}
        if value != "":
            body["value"] = value
        result = self._post(uri, body)
        logger.debug("Tuya IR command %s=%s → %s", code, value, result)
        return result

    def _send_ac_command_sync(
        self,
        power: bool | None = None,
        temp: int | None = None,
        mode: int | None = None,
        fan: int | None = None,
    ) -> str:
        """Send command to AC via IR hub (synchronous).

        Uses the Tuya combined AC endpoint (scenes/command) which sends a
        SINGLE IR signal containing the full AC state. This matches how
        physical AC remotes work — each IR blast carries power+temp+mode+fan.
        The SmartLife app uses this same endpoint.
        """
        # Update tracked state with any new values
        if power is not None:
            self._ac_state["power"] = "1" if power else "0"
        if temp is not None:
            self._ac_state["temp"] = str(temp)
        if mode is not None:
            self._ac_state["mode"] = str(mode)
        if fan is not None:
            self._ac_state["wind"] = str(fan)

        if power is None and temp is None and mode is None and fan is None:
            return "Tidak ada perintah AC yang diberikan."

        uri = f"infrareds/{_IR_HUB_ID}/air-conditioners/{_AC_REMOTE_ID}/scenes/command"
        payload = self._ac_state.copy()
        logger.debug("AC scenes/command payload: %s", payload)
        resp = self._post(uri, payload)
        success = resp.get("success", False) if isinstance(resp, dict) else False

        if not success:
            logger.warning("AC combined command failed: %s", resp)
            return f"Gagal mengontrol AC: {resp}"

        # Build human-readable result
        results = []
        if power is not None:
            results.append("dinyalakan" if power else "dimatikan")
        if temp is not None:
            results.append(f"suhu {temp}°C")
        if mode is not None:
            mode_name = AC_MODES.get(mode, str(mode))
            results.append(f"mode {mode_name}")
        if fan is not None:
            fan_name = AC_FAN_SPEEDS.get(fan, str(fan))
            results.append(f"kipas {fan_name}")

        return "AC " + ", ".join(results) + "."

    def _send_tv_ir_command_sync(self, command: str) -> str:
        """Send IR command to TV (synchronous).

        Uses the Tuya raw key command endpoint:
        POST /v2.0/infrareds/{infrared_id}/remotes/{remote_id}/raw/command
        with body {"category_id": 2, "key": "<key_name>"}.
        """
        uri = f"infrareds/{_IR_HUB_ID}/remotes/{_TV_REMOTE_ID}/raw/command"
        key = _TV_KEY_MAP.get(command, command.lower())
        resp = self._post(uri, {"category_id": _TV_CATEGORY_ID, "key": key})
        success = resp.get("success", False) if isinstance(resp, dict) else False
        if success:
            logger.info("TV IR command sent: %s (key=%s)", command, key)
            return f"Perintah TV IR '{command}' berhasil dikirim."
        logger.warning("TV IR command failed: %s (key=%s) → %s", command, key, resp)
        return f"Gagal mengirim perintah TV IR '{command}'."

    async def send_ac_command(
        self,
        power: bool | None = None,
        temp: int | None = None,
        mode: int | None = None,
        fan: int | None = None,
    ) -> str:
        """Send command to AC via IR hub (async wrapper).

        Args:
            power: True to turn on, False to turn off, None to skip.
            temp: Temperature 16-30, or None to skip.
            mode: 0=cool, 1=heat, 2=auto, 3=fan, 4=dry, or None.
            fan: 0=auto, 1=low, 2=medium, 3=high, or None.

        Returns:
            Human-readable status string.
        """
        return await asyncio.to_thread(
            self._send_ac_command_sync, power, temp, mode, fan,
        )

    async def send_tv_ir_command(self, command: str) -> str:
        """Send IR command to TV (async wrapper).

        Args:
            command: IR key name — Power, Volume+, Volume-, Channel+,
                     Channel-, Up, Down, Left, Right, OK, Home, Back, Menu,
                     or digit 0-9.

        Returns:
            Human-readable status string.
        """
        return await asyncio.to_thread(self._send_tv_ir_command_sync, command)
