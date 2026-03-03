"""LG WebOS TV driver — controls LG TVs over LAN via aiowebostv.

Create one instance per TV. First connection requires user to accept
the pairing popup on the TV screen. After that, the client_key is saved
for automatic reconnection.
"""

import asyncio
import logging

from aiowebostv import WebOsClient

logger = logging.getLogger(__name__)

# Common LG app IDs
LG_APPS: dict[str, str] = {
    "youtube": "youtube.leanback.v4",
    "netflix": "netflix",
    "disney": "com.disney.disneyplus-prod",
    "spotify": "spotify-beehive",
    "browser": "com.webos.app.browser",
    "hdmi1": "com.webos.app.hdmi1",
    "hdmi2": "com.webos.app.hdmi2",
    "hdmi3": "com.webos.app.hdmi3",
    "tv": "com.webos.app.livetv",
    "amazon": "amazon",
    "prime": "amazon",
}

# Connection timeout in seconds
_CONNECT_TIMEOUT = 5


class LGWebOSDriver:
    """Controls an LG WebOS TV via local network.

    Usage:
        tv = LGWebOSDriver(ip="192.168.1.100", name="tv_atas")
        await tv.connect()
        await tv.launch_app("youtube")
    """

    def __init__(
        self,
        ip: str,
        name: str = "tv",
        client_key: str | None = None,
    ) -> None:
        self._ip = ip
        self._name = name
        self._client_key = client_key or None
        self._client: WebOsClient | None = None

    @property
    def name(self) -> str:
        return self._name

    @property
    def client_key(self) -> str | None:
        return self._client_key

    async def connect(self, timeout: float | None = None) -> bool:
        """Connect to TV. First time requires user to accept on TV screen.

        Args:
            timeout: Connection timeout in seconds. Defaults to
                     _CONNECT_TIMEOUT (5s). Use longer for first pairing.

        Returns:
            True if connected successfully.

        Raises:
            ConnectionError: If TV is unreachable or refuses connection.
        """
        if timeout is None:
            timeout = _CONNECT_TIMEOUT
        try:
            self._client = WebOsClient(self._ip, client_key=self._client_key)
            await asyncio.wait_for(
                self._client.connect(), timeout=timeout,
            )
        except (asyncio.TimeoutError, OSError, ConnectionRefusedError) as e:
            self._client = None
            raise ConnectionError(
                f"Tidak bisa terhubung ke {self._name} ({self._ip}): {e}"
            ) from e

        # Save client_key after first pairing
        if self._client.client_key and self._client.client_key != self._client_key:
            self._client_key = self._client.client_key
            logger.info(
                "LG TV %s paired, client_key: %s", self._name, self._client_key,
            )
        logger.info("Connected to LG TV %s (%s)", self._name, self._ip)
        return True

    async def disconnect(self) -> None:
        """Disconnect from TV."""
        if self._client:
            await self._client.disconnect()
            self._client = None

    async def _ensure_connected(self) -> None:
        """Reconnect if not currently connected."""
        if self._client is None or not self._client.is_connected():
            await self.connect()

    async def launch_app(self, app_name: str) -> str:
        """Launch an app by friendly name.

        Args:
            app_name: Friendly name like 'youtube', 'netflix', 'spotify'.

        Returns:
            Status message.
        """
        app_id = LG_APPS.get(app_name.lower())
        if not app_id:
            available = ", ".join(sorted(LG_APPS.keys()))
            return f"App '{app_name}' tidak dikenali. Yang tersedia: {available}."
        await self._ensure_connected()
        await self._client.launch_app(app_id)
        logger.info("LG TV %s → launched %s", self._name, app_name)
        return f"Membuka {app_name} di {self._name}."

    async def set_volume(self, level: int) -> str:
        """Set volume level.

        Args:
            level: Volume 0-100.

        Returns:
            Status message.
        """
        await self._ensure_connected()
        await self._client.set_volume(level)
        logger.info("LG TV %s → volume %d", self._name, level)
        return f"Volume {self._name} diset ke {level}."

    async def volume_up(self) -> str:
        """Increase volume by one step."""
        await self._ensure_connected()
        await self._client.volume_up()
        logger.info("LG TV %s → volume up", self._name)
        return f"Volume {self._name} dinaikkan."

    async def volume_down(self) -> str:
        """Decrease volume by one step."""
        await self._ensure_connected()
        await self._client.volume_down()
        logger.info("LG TV %s → volume down", self._name)
        return f"Volume {self._name} diturunkan."

    async def power_off(self) -> str:
        """Turn off TV via WebOS.

        Note: WebOS cannot power ON a sleeping TV — use IR for that.

        Returns:
            Status message.
        """
        await self._ensure_connected()
        await self._client.power_off()
        self._client = None  # Connection drops after power off
        logger.info("LG TV %s → powered off", self._name)
        return f"{self._name} dimatikan."

    async def get_volume(self) -> int | None:
        """Get current volume level, or None if unavailable."""
        try:
            await self._ensure_connected()
            vol_info = await self._client.get_volume()
            return vol_info.get("volume") if isinstance(vol_info, dict) else None
        except Exception:
            return None
