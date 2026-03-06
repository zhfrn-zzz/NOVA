"""WebSocket server for remote agent connections.

Accepts connections from remote agents (e.g. Windows laptop) and routes
tool calls to them.  When no agent is connected, tools execute locally
as usual.

Protocol (JSON over WebSocket):

    Agent → Server:  {"type":"register", "device":"<name>", "token":"<secret>"}
    Server → Agent:  {"type":"registered"} | {"type":"error","message":"..."}
    Server → Agent:  {"type":"tool_call","id":"<uuid>","name":"...","args":{}}
    Agent → Server:  {"type":"tool_result","id":"<uuid>","result":"..."}
    Server → Agent:  {"type":"ping"}
    Agent → Server:  {"type":"pong"}
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid

import websockets
from websockets.asyncio.server import ServerConnection

from nova.config import get_config

logger = logging.getLogger(__name__)

REMOTE_TOOLS: set[str] = {
    # System control
    "volume_up", "volume_down", "mute_unmute",
    "play_pause_media", "next_track", "previous_track",
    "open_app", "open_browser", "open_url", "open_terminal", "open_file_manager",
    "lock_screen", "shutdown_pc", "restart_pc", "sleep_pc",
    "take_screenshot", "set_timer",
    # Music player (local only — TV targets handled server-side)
    "play_music",
    "pause_resume_music", "skip_track", "previous_music_track", "stop_music",
    # Dictation
    "dictate",
    # Display
    "brightness_up", "brightness_down", "get_brightness",
    # Network
    "wifi_on", "wifi_off", "get_wifi_status",
    # System info
    "get_battery_level", "get_ram_usage", "get_storage_info",
    "get_ip_address", "get_system_uptime",
}


def should_route_remote(name: str, args: dict | None = None) -> bool:
    """Check whether a tool call should be routed to the remote agent.

    Args:
        name: Tool function name.
        args: Tool arguments.

    Returns:
        True if this tool should be forwarded to the remote agent.
    """
    if name not in REMOTE_TOOLS:
        return False
    # play_music: only route to agent when target is "local" (default)
    if name == "play_music":
        return (args or {}).get("target", "local") == "local"
    return True


class RemoteAgentServer:
    """WebSocket server that manages a single remote agent connection."""

    def __init__(self) -> None:
        self._agent: ServerConnection | None = None
        self._agent_device: str = ""
        self._pending: dict[str, asyncio.Future[str]] = {}
        self._server: websockets.asyncio.server.Server | None = None
        self._keepalive_task: asyncio.Task | None = None

    @property
    def has_agent(self) -> bool:
        """True if a remote agent is currently connected."""
        return self._agent is not None

    @property
    def agent_device(self) -> str:
        """Name of the connected agent device."""
        return self._agent_device

    async def start(self, port: int) -> None:
        """Start listening for agent connections on the given port."""
        self._server = await websockets.asyncio.server.serve(
            self._handler,
            "0.0.0.0",
            port,
        )
        logger.info("Remote agent server listening on ws://0.0.0.0:%d", port)

    async def stop(self) -> None:
        """Shut down the server and disconnect any agent."""
        if self._keepalive_task:
            self._keepalive_task.cancel()
        if self._server:
            self._server.close()
            await self._server.wait_closed()
        self._agent = None
        for fut in self._pending.values():
            fut.cancel()
        self._pending.clear()

    async def execute_remote(self, name: str, args: dict | None = None) -> str:
        """Send a tool call to the remote agent and wait for the result.

        Args:
            name: Tool function name.
            args: Tool arguments dict.

        Returns:
            The tool result string from the remote agent.

        Raises:
            ConnectionError: If no agent is connected.
            TimeoutError: If the agent doesn't respond in time.
        """
        if self._agent is None:
            raise ConnectionError("No remote agent connected")

        call_id = str(uuid.uuid4())
        future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
        self._pending[call_id] = future

        msg = json.dumps({
            "type": "tool_call",
            "id": call_id,
            "name": name,
            "args": args or {},
        })

        try:
            await self._agent.send(msg)
            logger.info("Remote tool_call sent: %s(%s) id=%s", name, args or "", call_id)

            from nova.tools.registry import get_tool_timeout
            timeout = get_tool_timeout(name)
            result = await asyncio.wait_for(future, timeout=timeout)
            return result
        except TimeoutError:
            logger.error("Remote tool %s timed out", name)
            self._pending.pop(call_id, None)
            return f"Remote agent timeout saat menjalankan {name}."
        except websockets.ConnectionClosed:
            logger.error("Remote agent disconnected during tool call %s", name)
            self._pending.pop(call_id, None)
            self._agent = None
            self._agent_device = ""
            return f"Remote agent terputus saat menjalankan {name}."

    # ── WebSocket handler ─────────────────────────────────────────────

    async def _handler(self, ws: ServerConnection) -> None:
        """Handle a new WebSocket connection from an agent."""
        remote = ws.remote_address
        logger.info("New WebSocket connection from %s", remote)

        try:
            # First message must be registration
            raw = await asyncio.wait_for(ws.recv(), timeout=10.0)
            msg = json.loads(raw)

            if msg.get("type") != "register":
                await ws.send(json.dumps({"type": "error", "message": "Expected register"}))
                await ws.close()
                return

            # Validate auth token
            config = get_config()
            if config.remote_agent_token:
                if msg.get("token") != config.remote_agent_token:
                    logger.warning("Remote agent auth failed from %s", remote)
                    await ws.send(json.dumps({"type": "error", "message": "Invalid token"}))
                    await ws.close()
                    return

            # Disconnect previous agent if any
            if self._agent is not None:
                logger.info("Replacing previous agent %s", self._agent_device)
                try:
                    await self._agent.close()
                except Exception:
                    pass

            self._agent = ws
            self._agent_device = msg.get("device", "unknown")
            logger.info("Remote agent registered: %s from %s", self._agent_device, remote)
            await ws.send(json.dumps({"type": "registered"}))

            # Start keepalive
            if self._keepalive_task:
                self._keepalive_task.cancel()
            self._keepalive_task = asyncio.create_task(self._keepalive(ws))

            # Listen for messages
            async for raw_msg in ws:
                try:
                    data = json.loads(raw_msg)
                except json.JSONDecodeError:
                    logger.warning("Invalid JSON from agent: %s", raw_msg[:100])
                    continue

                msg_type = data.get("type")

                if msg_type == "tool_result":
                    call_id = data.get("id")
                    future = self._pending.pop(call_id, None)
                    if future and not future.done():
                        error = data.get("error")
                        if error:
                            future.set_result(f"Remote agent error: {error}")
                        else:
                            future.set_result(data.get("result", ""))
                    else:
                        logger.warning("Unexpected tool_result id=%s", call_id)

                elif msg_type == "pong":
                    pass  # keepalive response

                else:
                    logger.debug("Unknown message type from agent: %s", msg_type)

        except websockets.ConnectionClosed:
            logger.info("Remote agent disconnected: %s", self._agent_device)
        except Exception:
            logger.exception("Error in agent handler")
        finally:
            if self._agent is ws:
                self._agent = None
                self._agent_device = ""
                # Cancel all pending calls
                for fut in self._pending.values():
                    if not fut.done():
                        fut.set_result("Remote agent terputus.")
                self._pending.clear()
                logger.info("Remote agent cleaned up")

    async def _keepalive(self, ws: ServerConnection) -> None:
        """Send periodic pings to keep the WebSocket alive."""
        try:
            while True:
                await asyncio.sleep(30)
                if self._agent is ws:
                    try:
                        await ws.send(json.dumps({"type": "ping"}))
                    except websockets.ConnectionClosed:
                        break
                else:
                    break
        except asyncio.CancelledError:
            pass


# ── Singleton ─────────────────────────────────────────────────────────

_server_instance: RemoteAgentServer | None = None


def get_remote_server() -> RemoteAgentServer | None:
    """Get the singleton RemoteAgentServer instance, or None if not started."""
    return _server_instance


async def start_remote_server() -> RemoteAgentServer:
    """Create and start the global RemoteAgentServer.

    Returns:
        The running server instance.
    """
    global _server_instance
    if _server_instance is not None:
        return _server_instance

    config = get_config()
    _server_instance = RemoteAgentServer()
    await _server_instance.start(config.remote_agent_port)
    return _server_instance


async def stop_remote_server() -> None:
    """Stop the global RemoteAgentServer if running."""
    global _server_instance
    if _server_instance:
        await _server_instance.stop()
        _server_instance = None
