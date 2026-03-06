"""WhatsApp bridge client — communicates with wa-bridge Node.js service.

Architecture:
1. NOVA starts a small HTTP server (port 3002) to receive incoming messages.
2. Registers this callback URL with wa-bridge.
3. When wa-bridge receives a WhatsApp message, it POSTs to NOVA's callback.
4. NOVA processes the message through orchestrator and returns the reply.
"""

import logging

import httpx
from aiohttp import web

logger = logging.getLogger(__name__)

WA_BRIDGE_URL = "http://localhost:3001"


class NovaWhatsAppClient:
    """Client that connects NOVA to the WhatsApp bridge.

    Runs a local HTTP callback server and communicates with
    the wa-bridge Node.js microservice over HTTP.
    """

    def __init__(
        self,
        orchestrator,  # noqa: ANN001
        allowed_numbers: list[str] | None = None,
        port: int = 3002,
    ) -> None:
        """Initialize the WhatsApp client.

        Args:
            orchestrator: NOVA Orchestrator instance.
            allowed_numbers: Authorized phone numbers (628xxx format).
            port: Local port for the callback HTTP server.
        """
        self._orchestrator = orchestrator
        self._allowed = set(allowed_numbers or [])
        self._port = port
        self._app = web.Application()
        self._runner: web.AppRunner | None = None
        self._setup_routes()

    def _setup_routes(self) -> None:
        """Register HTTP routes for incoming messages."""
        self._app.router.add_post("/incoming", self._handle_incoming)

    async def _handle_incoming(self, request: web.Request) -> web.Response:
        """Handle incoming WhatsApp message forwarded by wa-bridge.

        Args:
            request: HTTP POST with JSON body {sender, text, jid}.

        Returns:
            JSON response with {reply: str}.
        """
        try:
            data = await request.json()
            sender = data.get("sender", "")
            text = data.get("text", "")

            if not text.strip():
                return web.json_response({"reply": ""})

            # Authorization (double-check, bridge also checks)
            if self._allowed and sender not in self._allowed:
                logger.warning("Unauthorized WhatsApp message from %s", sender)
                return web.json_response({"reply": ""})

            logger.info("WhatsApp message from %s: '%s'", sender, text)

            response = await self._orchestrator.handle_interaction(
                text,
                mode="text",
            )

            return web.json_response({"reply": response})

        except Exception:
            logger.exception("Error processing WhatsApp message")
            return web.json_response(
                {"reply": "Terjadi kesalahan saat memproses pesan."}
            )

    async def start(self) -> None:
        """Start the callback HTTP server and register with wa-bridge."""
        self._runner = web.AppRunner(self._app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", self._port)
        await site.start()
        logger.info("WhatsApp callback server running on port %d", self._port)

        await self._register_callback()

    async def _register_callback(self) -> None:
        """Tell wa-bridge where to send incoming messages."""
        callback_url = f"http://localhost:{self._port}/incoming"
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{WA_BRIDGE_URL}/register-callback",
                    json={"url": callback_url},
                    timeout=5,
                )
                if resp.status_code == 200:
                    logger.info("Registered callback with wa-bridge: %s", callback_url)
                else:
                    logger.error("Failed to register callback: %s", resp.text)
        except httpx.ConnectError:
            logger.warning(
                "wa-bridge not running at %s. WhatsApp integration disabled. "
                "Start it with: cd wa-bridge && npm start",
                WA_BRIDGE_URL,
            )
        except Exception:
            logger.exception("Error registering callback with wa-bridge")

    async def send_message(self, to: str, text: str) -> bool:
        """Send a message via WhatsApp (NOVA → user).

        Args:
            to: Phone number with country code, no + (e.g., "628123456789").
            text: Message text.

        Returns:
            True if message was sent successfully.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{WA_BRIDGE_URL}/send",
                    json={"to": to, "text": text},
                    timeout=10,
                )
                return resp.status_code == 200
        except Exception:
            logger.exception("Failed to send WhatsApp message")
            return False

    async def check_health(self) -> bool:
        """Check if wa-bridge is running and WhatsApp is connected.

        Returns:
            True if bridge is up and WhatsApp is connected.
        """
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.get(f"{WA_BRIDGE_URL}/health", timeout=3)
                data = resp.json()
                return data.get("connected", False)
        except Exception:
            return False

    async def stop(self) -> None:
        """Stop the callback server."""
        if self._runner:
            await self._runner.cleanup()
