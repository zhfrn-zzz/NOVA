"""NOVA Remote Agent — Windows system tray application that connects to the
NOVA server via WebSocket and executes system-control tools locally.

Features:
  - System tray icon with connection status (green/red/yellow)
  - Settings dialog for server URL, token, and device name
  - Auto-reconnect on disconnect
  - Balloon notifications for connection events
  - Config persisted to ~/.nova-agent/config.json
  - Can be compiled to single .exe with PyInstaller

Usage (from source):
    python nova_agent.py                   # GUI mode (tray icon)
    python nova_agent.py --console         # Console-only (no tray)
    python nova_agent.py --server ws://... # Override server URL
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import threading
import tkinter as tk
import webbrowser
from datetime import datetime
from pathlib import Path

import pystray
import websockets
from PIL import Image, ImageDraw

# ── Paths & Logging ──────────────────────────────────────────────────

APP_DIR = Path.home() / ".nova-agent"
APP_DIR.mkdir(exist_ok=True)
CONFIG_PATH = APP_DIR / "config.json"
LOG_PATH = APP_DIR / "agent.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(LOG_PATH, encoding="utf-8"),
    ],
)
logger = logging.getLogger("nova-agent")


# ── Config Persistence ───────────────────────────────────────────────

_DEFAULT_CONFIG = {
    "server_url": "",
    "token": "",
    "device_name": "windows-laptop",
}


def load_config() -> dict:
    """Load config from JSON file, merging with defaults."""
    config = dict(_DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, encoding="utf-8") as f:
                config.update(json.load(f))
        except (json.JSONDecodeError, OSError):
            logger.warning("Could not read config, using defaults")
    return config


def save_config(config: dict) -> None:
    """Save config to JSON file."""
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)
    logger.info("Config saved to %s", CONFIG_PATH)


# ── Tray Icon Generation ────────────────────────────────────────────

_COLORS = {
    "connected": "#22c55e",     # green
    "disconnected": "#ef4444",  # red
    "connecting": "#f59e0b",    # amber
}


def _create_icon_image(status: str) -> Image.Image:
    """Generate a 64x64 tray icon with status color and 'N' letter."""
    size = 64
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    color = _COLORS.get(status, _COLORS["disconnected"])
    draw.ellipse([4, 4, size - 4, size - 4], fill=color, outline="white", width=3)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype("arial.ttf", 32)
    except (OSError, ImportError):
        font = ImageFont.load_default()
    bbox = draw.textbbox((0, 0), "N", font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    draw.text(
        ((size - tw) / 2 - bbox[0], (size - th) / 2 - bbox[1]),
        "N", fill="white", font=font,
    )
    return img


# ── Settings Dialog (tkinter) ───────────────────────────────────────

def show_settings_dialog(current_config: dict) -> dict | None:
    """Show a modal settings window. Returns updated config or None."""
    result = {}

    root = tk.Tk()
    root.title("NOVA Agent — Settings")
    root.resizable(False, False)
    root.configure(bg="#1e1e2e")

    # Center on screen
    w, h = 420, 260
    sx = (root.winfo_screenwidth() - w) // 2
    sy = (root.winfo_screenheight() - h) // 2
    root.geometry(f"{w}x{h}+{sx}+{sy}")

    style = {"bg": "#1e1e2e", "fg": "#cdd6f4", "font": ("Segoe UI", 10)}
    entry_style = {
        "bg": "#313244", "fg": "#cdd6f4", "insertbackground": "#cdd6f4",
        "font": ("Segoe UI", 10), "relief": "flat", "highlightthickness": 1,
        "highlightcolor": "#89b4fa", "highlightbackground": "#45475a",
    }

    tk.Label(root, text="Server URL", **style).place(x=20, y=20)
    sv_entry = tk.Entry(root, width=38, **entry_style)
    sv_entry.place(x=20, y=45, height=30)
    sv_entry.insert(0, current_config.get("server_url", ""))

    tk.Label(root, text="Auth Token (opsional)", **style).place(x=20, y=85)
    tk_entry = tk.Entry(root, width=38, show="*", **entry_style)
    tk_entry.place(x=20, y=110, height=30)
    tk_entry.insert(0, current_config.get("token", ""))

    tk.Label(root, text="Device Name", **style).place(x=20, y=150)
    dn_entry = tk.Entry(root, width=38, **entry_style)
    dn_entry.place(x=20, y=175, height=30)
    dn_entry.insert(0, current_config.get("device_name", "windows-laptop"))

    def on_save():
        url = sv_entry.get().strip()
        if not url:
            sv_entry.configure(highlightcolor="#ef4444", highlightbackground="#ef4444")
            return
        result["server_url"] = url
        result["token"] = tk_entry.get().strip()
        result["device_name"] = dn_entry.get().strip() or "windows-laptop"
        root.destroy()

    def on_cancel():
        root.destroy()

    btn_style = {"font": ("Segoe UI", 10, "bold"), "relief": "flat", "cursor": "hand2"}
    tk.Button(
        root, text="Save", command=on_save,
        bg="#89b4fa", fg="#1e1e2e", activebackground="#74c7ec", **btn_style,
    ).place(x=200, y=215, width=90, height=32)
    tk.Button(
        root, text="Cancel", command=on_cancel,
        bg="#45475a", fg="#cdd6f4", activebackground="#585b70", **btn_style,
    ).place(x=310, y=215, width=90, height=32)

    root.protocol("WM_DELETE_WINDOW", on_cancel)
    sv_entry.focus_set()
    root.mainloop()
    return result if result else None


# =====================================================================
# Tool implementations (Windows-only) — unchanged from original
# =====================================================================

# ── Helpers ──────────────────────────────────────────────────────────

async def _send_key(vk_code: int) -> None:
    cmd = (
        'powershell -Command "'
        "$shell = New-Object -ComObject WScript.Shell; "
        f'$shell.SendKeys([char]{vk_code})"'
    )
    proc = await asyncio.create_subprocess_shell(
        cmd, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    await proc.wait()


def _popen(args: list[str]) -> None:
    subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


async def _run_cmd(args: list[str]) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return (
        proc.returncode or 0,
        stdout.decode("utf-8", errors="replace").strip(),
        stderr.decode("utf-8", errors="replace").strip(),
    )


# ── Volume / Media ───────────────────────────────────────────────────

async def volume_up(**_) -> str:
    await _send_key(175)
    await _send_key(175)
    return "Volume telah dinaikkan."


async def volume_down(**_) -> str:
    await _send_key(174)
    await _send_key(174)
    return "Volume telah diturunkan."


async def mute_unmute(**_) -> str:
    await _send_key(173)
    return "Mute telah di-toggle."


async def play_pause_media(**_) -> str:
    await _send_key(179)
    return "Media play/pause di-toggle."


async def next_track(**_) -> str:
    await _send_key(176)
    return "Berpindah ke track selanjutnya."


async def previous_track(**_) -> str:
    await _send_key(177)
    return "Berpindah ke track sebelumnya."


# ── Applications ─────────────────────────────────────────────────────

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


async def open_app(app_name: str = "", **_) -> str:
    key = app_name.strip().lower()
    cmd_args = _APP_MAP.get(key, ["cmd", "/c", "start", key])
    try:
        _popen(cmd_args)
        return f"{app_name} telah dibuka."
    except Exception as e:
        return f"Gagal membuka {app_name}: {e}"


async def open_browser(**_) -> str:
    try:
        _popen(["cmd", "/c", "start", "https://www.google.com"])
        return "Browser telah dibuka."
    except Exception as e:
        return f"Gagal membuka browser: {e}"


async def open_url(url: str = "", **_) -> str:
    try:
        _popen(["cmd", "/c", "start", url])
        return f"Membuka {url}."
    except Exception as e:
        return f"Gagal membuka URL: {e}"


async def open_terminal(**_) -> str:
    try:
        _popen(["wt"])
        return "Terminal telah dibuka."
    except Exception as e:
        return f"Gagal membuka terminal: {e}"


async def open_file_manager(**_) -> str:
    try:
        _popen(["explorer.exe"])
        return "File manager telah dibuka."
    except Exception as e:
        return f"Gagal membuka file manager: {e}"


# ── System Power ─────────────────────────────────────────────────────

async def lock_screen(**_) -> str:
    try:
        _popen(["rundll32.exe", "user32.dll,LockWorkStation"])
        return "Layar telah dikunci."
    except Exception as e:
        return f"Gagal mengunci layar: {e}"


async def shutdown_pc(delay_seconds: int = 60, **_) -> str:
    try:
        _popen(["shutdown", "/s", "/t", str(delay_seconds)])
        return f"Komputer akan dimatikan dalam {delay_seconds} detik."
    except Exception as e:
        return f"Gagal menjadwalkan shutdown: {e}"


async def restart_pc(delay_seconds: int = 60, **_) -> str:
    try:
        _popen(["shutdown", "/r", "/t", str(delay_seconds)])
        return f"Komputer akan di-restart dalam {delay_seconds} detik."
    except Exception as e:
        return f"Gagal menjadwalkan restart: {e}"


async def sleep_pc(**_) -> str:
    try:
        _popen([
            "powershell", "-Command",
            "Add-Type -AssemblyName System.Windows.Forms; "
            "[System.Windows.Forms.Application]::SetSuspendState("
            "'Suspend', $false, $false)",
        ])
        return "Komputer akan masuk mode sleep."
    except Exception as e:
        return f"Gagal sleep: {e}"


# ── Screenshot ───────────────────────────────────────────────────────

async def take_screenshot(**_) -> str:
    screenshots_dir = Path.home() / "Pictures" / "Screenshots"
    screenshots_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = screenshots_dir / f"screenshot_{timestamp}.png"
    try:
        ps_cmd = (
            "Add-Type -AssemblyName System.Windows.Forms; "
            "$screen = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds; "
            "$bitmap = New-Object System.Drawing.Bitmap("
            "$screen.Width, $screen.Height); "
            "$graphics = [System.Drawing.Graphics]::FromImage($bitmap); "
            "$graphics.CopyFromScreen($screen.Location, "
            "[System.Drawing.Point]::Empty, $screen.Size); "
            f"$bitmap.Save('{filepath}'); "
            "$graphics.Dispose(); $bitmap.Dispose()"
        )
        proc = await asyncio.create_subprocess_exec(
            "powershell", "-Command", ps_cmd,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            return f"Screenshot gagal: {stderr.decode().strip()}"
        return f"Screenshot tersimpan di {filepath}."
    except Exception as e:
        return f"Gagal mengambil screenshot: {e}"


# ── Timer ────────────────────────────────────────────────────────────

async def set_timer(seconds: int = 0, label: str = "Timer", **_) -> str:
    if seconds <= 0:
        return "Durasi timer harus lebih dari 0 detik."

    async def _timer_task() -> None:
        await asyncio.sleep(seconds)
        try:
            ps_cmd = (
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$n = New-Object System.Windows.Forms.NotifyIcon; "
                "$n.Icon = [System.Drawing.SystemIcons]::Information; "
                "$n.Visible = $true; "
                f"$n.ShowBalloonTip(5000, 'NOVA Timer', '{label}', "
                "'Info'); Start-Sleep -Seconds 6; $n.Dispose()"
            )
            proc = await asyncio.create_subprocess_exec(
                "powershell", "-Command", ps_cmd,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await proc.wait()
        except Exception:
            logger.debug("Timer notification failed", exc_info=True)

    asyncio.create_task(_timer_task())
    if seconds >= 60:
        mins = seconds // 60
        secs = seconds % 60
        time_str = f"{mins} menit" + (f" {secs} detik" if secs else "")
    else:
        time_str = f"{seconds} detik"
    return f"Timer {label} telah diset untuk {time_str}."


# ── Music Player ─────────────────────────────────────────────────────

_YTDLP_TIMEOUT = 10.0


async def _search_youtube(query: str) -> str | None:
    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "yt-dlp", f"ytsearch:{query}",
                "--get-id", "--no-warnings", "--no-playlist",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            ),
            timeout=_YTDLP_TIMEOUT,
        )
        stdout, _ = await proc.communicate()
        vid = stdout.decode("utf-8", errors="replace").strip()
        return vid.splitlines()[0].strip() if vid else None
    except (TimeoutError, FileNotFoundError):
        return None


async def play_music(query: str = "", **_) -> str:
    if not query.strip():
        return "Tidak ada lagu yang diminta."
    video_id = await _search_youtube(query)
    if not video_id:
        return f"Tidak menemukan lagu untuk: {query}"
    url = f"https://music.youtube.com/watch?v={video_id}"
    webbrowser.open(url)
    return f"Memutar lagu: {query} — {url}"


async def pause_resume_music(**_) -> str:
    try:
        import pyautogui
        await asyncio.to_thread(pyautogui.press, "playpause")
        return "Musik di-pause/resume."
    except ImportError:
        return "pyautogui belum terinstall."
    except Exception as e:
        return f"Gagal pause/resume musik: {e}"


async def skip_track(**_) -> str:
    try:
        import pyautogui
        await asyncio.to_thread(pyautogui.press, "nexttrack")
        return "Beralih ke lagu selanjutnya."
    except ImportError:
        return "pyautogui belum terinstall."
    except Exception as e:
        return f"Gagal skip lagu: {e}"


async def previous_music_track(**_) -> str:
    try:
        import pyautogui
        await asyncio.to_thread(pyautogui.press, "prevtrack")
        return "Kembali ke lagu sebelumnya."
    except ImportError:
        return "pyautogui belum terinstall."
    except Exception as e:
        return f"Gagal ke lagu sebelumnya: {e}"


async def stop_music(**_) -> str:
    try:
        import pyautogui
        await asyncio.to_thread(pyautogui.press, "stop")
        return "Musik dihentikan."
    except ImportError:
        return "pyautogui belum terinstall."
    except Exception as e:
        return f"Gagal menghentikan musik: {e}"


# ── Dictation ────────────────────────────────────────────────────────

async def dictate(text: str = "", **_) -> str:
    if not text.strip():
        return "Tidak ada teks untuk diketik."
    try:
        import pyautogui
        await asyncio.sleep(0.5)
        try:
            pyautogui.write(text, interval=0.02)
        except Exception:
            import pyperclip
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
        return (
            f"Teks berhasil diketik: "
            f"{text[:50]}{'...' if len(text) > 50 else ''}"
        )
    except ImportError:
        return "pyautogui belum terinstall."
    except Exception as e:
        return f"Gagal mengetik teks: {e}"


# ── Display / Brightness ─────────────────────────────────────────────

async def brightness_up(**_) -> str:
    try:
        import screen_brightness_control as sbc
        current = await asyncio.to_thread(sbc.get_brightness)
        level = current[0] if isinstance(current, list) else current
        new_level = min(level + 10, 100)
        await asyncio.to_thread(sbc.set_brightness, new_level)
        return f"Brightness dinaikkan ke {new_level}%."
    except ImportError:
        return "screen-brightness-control belum terinstall."
    except Exception as e:
        return f"Gagal menaikkan brightness: {e}"


async def brightness_down(**_) -> str:
    try:
        import screen_brightness_control as sbc
        current = await asyncio.to_thread(sbc.get_brightness)
        level = current[0] if isinstance(current, list) else current
        new_level = max(level - 10, 0)
        await asyncio.to_thread(sbc.set_brightness, new_level)
        return f"Brightness diturunkan ke {new_level}%."
    except ImportError:
        return "screen-brightness-control belum terinstall."
    except Exception as e:
        return f"Gagal menurunkan brightness: {e}"


async def get_brightness(**_) -> str:
    try:
        import screen_brightness_control as sbc
        current = await asyncio.to_thread(sbc.get_brightness)
        level = current[0] if isinstance(current, list) else current
        return f"Brightness saat ini: {level}%."
    except ImportError:
        return "screen-brightness-control belum terinstall."
    except Exception as e:
        return f"Gagal mendapatkan brightness: {e}"


# ── Network / Wi-Fi ──────────────────────────────────────────────────

async def wifi_on(**_) -> str:
    try:
        rc, _, err = await _run_cmd([
            "netsh", "interface", "set", "interface", "Wi-Fi", "enable",
        ])
        return "Wi-Fi telah diaktifkan." if rc == 0 else f"Gagal: {err}"
    except Exception as e:
        return f"Gagal mengaktifkan Wi-Fi: {e}"


async def wifi_off(**_) -> str:
    try:
        rc, _, err = await _run_cmd([
            "netsh", "interface", "set", "interface", "Wi-Fi", "disable",
        ])
        return "Wi-Fi telah dinonaktifkan." if rc == 0 else f"Gagal: {err}"
    except Exception as e:
        return f"Gagal menonaktifkan Wi-Fi: {e}"


async def get_wifi_status(**_) -> str:
    try:
        rc, stdout, _ = await _run_cmd(["netsh", "wlan", "show", "interfaces"])
        if rc != 0:
            return "Gagal mendapatkan status Wi-Fi."
        ssid = state = None
        for line in stdout.splitlines():
            line = line.strip()
            if line.startswith("SSID") and "BSSID" not in line:
                ssid = line.split(":", 1)[1].strip() if ":" in line else None
            if line.startswith("State"):
                state = line.split(":", 1)[1].strip() if ":" in line else None
        if state and "connected" in state.lower():
            return f"Wi-Fi terhubung ke: {ssid}."
        return "Wi-Fi tidak terhubung."
    except Exception as e:
        return f"Gagal mendapatkan status Wi-Fi: {e}"


# ── System Info ──────────────────────────────────────────────────────

async def get_battery_level(**_) -> str:
    try:
        import psutil
        battery = psutil.sensors_battery()
        if battery is None:
            return "Tidak ada baterai terdeteksi (kemungkinan PC desktop)."
        pct = round(battery.percent)
        charging = (
            "sedang mengisi (charging)" if battery.power_plugged
            else "tidak mengisi (discharging)"
        )
        return f"Baterai: {pct}%, {charging}."
    except Exception as e:
        return f"Gagal mendapatkan info baterai: {e}"


async def get_ram_usage(**_) -> str:
    try:
        import psutil
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024 ** 3)
        total_gb = mem.total / (1024 ** 3)
        return f"RAM: {used_gb:.1f} GB / {total_gb:.1f} GB ({mem.percent}% terpakai)."
    except Exception as e:
        return f"Gagal mendapatkan info RAM: {e}"


async def get_storage_info(**_) -> str:
    try:
        import psutil
        disk = psutil.disk_usage("C:\\")
        used_gb = disk.used / (1024 ** 3)
        total_gb = disk.total / (1024 ** 3)
        free_gb = disk.free / (1024 ** 3)
        return (
            f"Storage: {used_gb:.1f} GB / {total_gb:.1f} GB terpakai "
            f"({free_gb:.1f} GB tersisa, {disk.percent}%)."
        )
    except Exception as e:
        return f"Gagal mendapatkan info storage: {e}"


async def get_ip_address(**_) -> str:
    try:
        local_ip = socket.gethostbyname(socket.gethostname())
        if local_ip.startswith("127."):
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            try:
                s.connect(("8.8.8.8", 80))
                local_ip = s.getsockname()[0]
            finally:
                s.close()
        public_ip = "tidak tersedia"
        try:
            import urllib.request
            with urllib.request.urlopen("https://ifconfig.me/ip", timeout=5) as r:
                public_ip = r.read().decode().strip()
        except Exception:
            pass
        return f"IP lokal: {local_ip}, IP publik: {public_ip}."
    except Exception as e:
        return f"Gagal mendapatkan IP address: {e}"


async def get_system_uptime(**_) -> str:
    try:
        import psutil
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime = datetime.now() - boot_time
        total_seconds = int(uptime.total_seconds())
        hours = total_seconds // 3600
        minutes = (total_seconds % 3600) // 60
        if hours > 0:
            return f"Sistem sudah menyala selama {hours} jam {minutes} menit."
        return f"Sistem sudah menyala selama {minutes} menit."
    except Exception as e:
        return f"Gagal mendapatkan uptime: {e}"


# =====================================================================
# Tool dispatch table
# =====================================================================

TOOLS: dict[str, object] = {
    "volume_up": volume_up, "volume_down": volume_down,
    "mute_unmute": mute_unmute,
    "play_pause_media": play_pause_media,
    "next_track": next_track, "previous_track": previous_track,
    "open_app": open_app, "open_browser": open_browser,
    "open_url": open_url, "open_terminal": open_terminal,
    "open_file_manager": open_file_manager,
    "lock_screen": lock_screen,
    "shutdown_pc": shutdown_pc, "restart_pc": restart_pc, "sleep_pc": sleep_pc,
    "take_screenshot": take_screenshot, "set_timer": set_timer,
    "play_music": play_music,
    "pause_resume_music": pause_resume_music, "skip_track": skip_track,
    "previous_music_track": previous_music_track, "stop_music": stop_music,
    "dictate": dictate,
    "brightness_up": brightness_up, "brightness_down": brightness_down,
    "get_brightness": get_brightness,
    "wifi_on": wifi_on, "wifi_off": wifi_off, "get_wifi_status": get_wifi_status,
    "get_battery_level": get_battery_level, "get_ram_usage": get_ram_usage,
    "get_storage_info": get_storage_info, "get_ip_address": get_ip_address,
    "get_system_uptime": get_system_uptime,
}


async def _execute(name: str, args: dict) -> str:
    impl = TOOLS.get(name)
    if impl is None:
        return f"Unknown tool: {name}"
    try:
        return await impl(**args)
    except Exception as e:
        logger.exception("Tool %s failed", name)
        return f"Error: {e}"


# =====================================================================
# WebSocket client with auto-reconnect
# =====================================================================

class NovaAgentClient:
    """Async WebSocket client that connects to NOVA server."""

    def __init__(
        self,
        server_url: str,
        token: str = "",
        device: str = "windows-laptop",
        on_status: object = None,
    ) -> None:
        self.server_url = server_url
        self.token = token
        self.device = device
        self._on_status = on_status  # callback(status: str, detail: str)
        self._stop_flag = threading.Event()  # thread-safe stop signal

    def _notify(self, status: str, detail: str = "") -> None:
        if self._on_status:
            try:
                self._on_status(status, detail)
            except Exception:
                pass

    async def run(self) -> None:
        """Main loop: connect → register → listen, auto-reconnect."""
        hostname = socket.gethostname()

        while not self._stop_flag.is_set():
            self._notify("connecting", f"Menghubungkan ke {self.server_url}...")
            try:
                async with websockets.connect(self.server_url) as ws:
                    reg = {"type": "register", "device": self.device,
                           "hostname": hostname}
                    if self.token:
                        reg["token"] = self.token
                    await ws.send(json.dumps(reg))

                    resp = json.loads(await ws.recv())
                    if resp.get("type") == "error":
                        self._notify("disconnected", resp.get("message", ""))
                        logger.error("Registration rejected: %s", resp.get("message"))
                        await asyncio.sleep(10)
                        continue

                    self._notify("connected", self.server_url)
                    logger.info("Connected to %s as '%s'", self.server_url, self.device)

                    async for raw in ws:
                        if self._stop_flag.is_set():
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue

                        if msg.get("type") == "tool_call":
                            call_id = msg["id"]
                            name = msg["name"]
                            args = msg.get("args", {})
                            logger.info("Exec: %s(%s)", name, args or "")
                            result = await _execute(name, args)
                            logger.info("Result: %s", result[:120])
                            await ws.send(json.dumps({
                                "type": "tool_result",
                                "id": call_id,
                                "result": result,
                            }))
                        elif msg.get("type") == "ping":
                            await ws.send(json.dumps({"type": "pong"}))

            except (websockets.ConnectionClosed, ConnectionRefusedError, OSError) as e:
                self._notify("disconnected", str(e))
                logger.warning("Disconnected (%s), reconnecting in 5s...", e)
            except Exception:
                self._notify("disconnected", "Unexpected error")
                logger.exception("Unexpected error, reconnecting in 5s...")

            if not self._stop_flag.is_set():
                await asyncio.sleep(5)

    def request_stop(self) -> None:
        self._stop_flag.set()


# =====================================================================
# System Tray Application
# =====================================================================

class NovaTrayApp:
    """System tray application that manages the WebSocket client."""

    def __init__(self, config: dict) -> None:
        self._config = config
        self._status = "disconnected"
        self._status_detail = ""
        self._icon: pystray.Icon | None = None
        self._client: NovaAgentClient | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ws_thread: threading.Thread | None = None

    def run(self) -> None:
        """Start the tray icon (blocks until Exit is clicked)."""
        self._icon = pystray.Icon(
            "nova-agent",
            icon=_create_icon_image("disconnected"),
            title="NOVA Agent — Disconnected",
            menu=self._build_menu(),
        )
        self._icon.run(setup=self._on_setup)

    # ── Menu ─────────────────────────────────────────────────────────

    def _build_menu(self) -> pystray.Menu:
        return pystray.Menu(
            pystray.MenuItem(
                lambda _: f"Status: {self._status_label()}",
                None, enabled=False,
            ),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Settings...", self._on_settings),
            pystray.MenuItem("View Logs", self._on_view_logs),
            pystray.MenuItem("Reconnect", self._on_reconnect),
            pystray.Menu.SEPARATOR,
            pystray.MenuItem("Exit", self._on_exit),
        )

    def _status_label(self) -> str:
        if self._status == "connected":
            return f"Connected to {self._status_detail}"
        if self._status == "connecting":
            return "Connecting..."
        return "Disconnected"

    # ── Callbacks ────────────────────────────────────────────────────

    def _on_setup(self, icon: pystray.Icon) -> None:
        """Called by pystray after the icon is ready (runs in background thread)."""
        icon.visible = True
        self._start_ws_thread()

    def _on_status_change(self, status: str, detail: str = "") -> None:
        """Called from the asyncio thread when connection status changes."""
        prev = self._status
        self._status = status
        self._status_detail = detail

        if self._icon:
            self._icon.icon = _create_icon_image(status)
            self._icon.title = f"NOVA Agent — {self._status_label()}"
            self._icon.update_menu()

            if status != prev:
                if status == "connected":
                    self._icon.notify("Terhubung ke NOVA server", "NOVA Agent")
                elif status == "disconnected" and prev == "connected":
                    self._icon.notify("Terputus dari NOVA server", "NOVA Agent")

    def _on_settings(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        new_cfg = show_settings_dialog(self._config)
        if new_cfg:
            self._config.update(new_cfg)
            save_config(self._config)
            self._restart_ws()

    def _on_view_logs(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        os.startfile(str(LOG_PATH))  # noqa: S606

    def _on_reconnect(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._restart_ws()

    def _on_exit(self, icon: pystray.Icon, item: pystray.MenuItem) -> None:
        self._stop_ws()
        icon.stop()

    # ── WebSocket thread management ──────────────────────────────────

    def _start_ws_thread(self) -> None:
        if not self._config.get("server_url"):
            logger.warning("No server URL configured")
            return

        self._loop = asyncio.new_event_loop()
        self._client = NovaAgentClient(
            server_url=self._config["server_url"],
            token=self._config.get("token", ""),
            device=self._config.get("device_name", "windows-laptop"),
            on_status=self._on_status_change,
        )

        def _run():
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._client.run())

        self._ws_thread = threading.Thread(target=_run, daemon=True)
        self._ws_thread.start()
        logger.info("WebSocket thread started")

    def _stop_ws(self) -> None:
        if self._client:
            self._client.request_stop()
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)

    def _restart_ws(self) -> None:
        self._stop_ws()
        if self._ws_thread and self._ws_thread.is_alive():
            self._ws_thread.join(timeout=3)
        self._start_ws_thread()


# =====================================================================
# Console-only mode (no GUI, for debugging or headless use)
# =====================================================================

def run_console(config: dict) -> None:
    """Run the agent in console-only mode (no system tray)."""
    def on_status(status: str, detail: str = "") -> None:
        logger.info("[%s] %s", status.upper(), detail)

    client = NovaAgentClient(
        server_url=config["server_url"],
        token=config.get("token", ""),
        device=config.get("device_name", "windows-laptop"),
        on_status=on_status,
    )
    try:
        asyncio.run(client.run())
    except KeyboardInterrupt:
        logger.info("Agent stopped.")


# =====================================================================
# Entry point
# =====================================================================

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="nova-agent",
        description="NOVA Remote Agent — system tray app for remote PC control",
    )
    parser.add_argument(
        "--server", "-s", default=None,
        help="WebSocket URL (overrides saved config)",
    )
    parser.add_argument(
        "--token", "-t", default=None,
        help="Auth token (overrides saved config)",
    )
    parser.add_argument(
        "--device", "-d", default=None,
        help="Device name (default: windows-laptop)",
    )
    parser.add_argument(
        "--console", action="store_true",
        help="Console-only mode (no system tray GUI)",
    )
    args = parser.parse_args()

    config = load_config()

    # CLI args override saved config
    if args.server:
        config["server_url"] = args.server
    if args.token is not None:
        config["token"] = args.token
    if args.device:
        config["device_name"] = args.device

    # If no server URL, show settings dialog
    if not config.get("server_url"):
        if args.console:
            print("ERROR: --server is required in console mode.")
            return
        new_cfg = show_settings_dialog(config)
        if not new_cfg:
            return
        config.update(new_cfg)
        save_config(config)

    logger.info("NOVA Remote Agent starting...")
    logger.info("Server: %s | Device: %s | Tools: %d",
                config["server_url"], config.get("device_name"), len(TOOLS))

    if args.console:
        run_console(config)
    else:
        app = NovaTrayApp(config)
        app.run()


if __name__ == "__main__":
    main()
