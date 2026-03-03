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
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load .env so TUYA_* vars are available via os.environ
load_dotenv(Path(__file__).resolve().parents[3] / ".env")

# Device IDs
_IR_HUB_ID = "a37d9677a6a3498269xabd"
_AC_REMOTE_ID = "a387dddea7a6953cf4yslm"
_TV_REMOTE_ID = "a31c92a675b6ccc8504k2v"

# AC mode mapping
AC_MODES = {0: "cool", 1: "heat", 2: "auto", 3: "fan", 4: "dry"}
AC_FAN_SPEEDS = {0: "auto", 1: "low", 2: "medium", 3: "high"}


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

    def _get_cloud(self) -> tinytuya.Cloud:
        """Lazy-init the tinytuya Cloud client."""
        if self._cloud is None:
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
        """Send command to AC via IR hub (synchronous)."""
        results = []
        uri = f"infrareds/{_IR_HUB_ID}/remotes/{_AC_REMOTE_ID}/command"

        if power is not None:
            code = "PowerOn" if power else "PowerOff"
            resp = self._post(uri, {"code": code})
            success = resp.get("success", False) if isinstance(resp, dict) else False
            if success:
                results.append("dinyalakan" if power else "dimatikan")
            else:
                logger.warning("AC power command failed: %s", resp)
                return f"Gagal {'menyalakan' if power else 'mematikan'} AC."

        if temp is not None:
            resp = self._post(uri, {"code": "T", "value": temp})
            success = resp.get("success", False) if isinstance(resp, dict) else False
            if success:
                results.append(f"suhu {temp}°C")
            else:
                logger.warning("AC temp command failed: %s", resp)
                return f"Gagal mengatur suhu AC ke {temp}°C."

        if mode is not None:
            resp = self._post(uri, {"code": "M", "value": mode})
            success = resp.get("success", False) if isinstance(resp, dict) else False
            if success:
                mode_name = AC_MODES.get(mode, str(mode))
                results.append(f"mode {mode_name}")
            else:
                logger.warning("AC mode command failed: %s", resp)
                return "Gagal mengatur mode AC."

        if fan is not None:
            resp = self._post(uri, {"code": "F", "value": fan})
            success = resp.get("success", False) if isinstance(resp, dict) else False
            if success:
                fan_name = AC_FAN_SPEEDS.get(fan, str(fan))
                results.append(f"kipas {fan_name}")
            else:
                logger.warning("AC fan command failed: %s", resp)
                return "Gagal mengatur kecepatan kipas AC."

        if not results:
            return "Tidak ada perintah AC yang diberikan."

        return "AC " + ", ".join(results) + "."

    def _send_tv_ir_command_sync(self, command: str) -> str:
        """Send IR command to TV (synchronous)."""
        uri = f"infrareds/{_IR_HUB_ID}/remotes/{_TV_REMOTE_ID}/command"
        resp = self._post(uri, {"code": command})
        success = resp.get("success", False) if isinstance(resp, dict) else False
        if success:
            logger.info("TV IR command sent: %s", command)
            return f"Perintah TV IR '{command}' berhasil dikirim."
        logger.warning("TV IR command failed: %s → %s", command, resp)
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
