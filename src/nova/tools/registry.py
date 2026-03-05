"""Tool registry — central catalogue of all NOVA tools for LLM function calling.

Defines FunctionDeclaration schemas for Gemini and dispatches tool calls
to the correct implementation.
"""

import logging

from google.genai import types

from nova.memory.persistent import (
    memory_forget,
    memory_search,
    memory_store,
    recall_facts,
    remember_fact,
    update_user_profile,
)
from nova.tools import (
    dictation,
    display_control,
    heartbeat_reminders,
    iot,
    music_player,
    network_control,
    notes,
    system_control,
    system_info,
    time_date,
    weather,
    web_search,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Function declarations for the Gemini function-calling API
# ---------------------------------------------------------------------------

_FUNCTION_DECLARATIONS = [
    # ── Time / Date ──────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="get_current_time",
        description=(
            "Get the current local time. Use this when the user asks what time it is, "
            "jam berapa, or any time-related question."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_current_date",
        description=(
            "Get the current local date. Use this when the user asks what today's date is, "
            "tanggal berapa, or any date-related question."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_current_datetime",
        description=(
            "Get both the current local date and time together."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Volume ───────────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="volume_up",
        description=(
            "Increase the system volume. Use when the user says 'volume up', "
            "'naikkan volume', 'louder', 'kerasin', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="volume_down",
        description=(
            "Decrease the system volume. Use when the user says 'volume down', "
            "'kecilkan volume', 'turunkan volume', 'quieter', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="mute_unmute",
        description=(
            "Toggle system audio mute/unmute. Use when the user says 'mute', "
            "'unmute', 'bisukan', 'matikan suara', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Media Controls ───────────────────────────────────────────────
    types.FunctionDeclaration(
        name="play_pause_media",
        description=(
            "Toggle media play/pause. Use when the user says 'play', 'pause', "
            "'putar musik', 'pause musik', 'stop musik', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="next_track",
        description=(
            "Skip to next media track. Use when the user says 'next', 'skip', "
            "'lagu selanjutnya', 'next track', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="previous_track",
        description=(
            "Go to previous media track. Use when the user says 'previous', 'back', "
            "'lagu sebelumnya', 'previous track', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Applications ─────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="open_app",
        description=(
            "Open an application by name. Supports: notepad, calculator, spotify, "
            "discord, whatsapp, vscode, explorer, paint, settings, task manager. "
            "Use when the user says 'buka spotify', 'open notepad', 'buka kalkulator', "
            "etc. For unlisted apps, try the name directly."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "app_name": {
                    "type": "string",
                    "description": "App name, e.g. 'spotify', 'notepad', 'vscode'.",
                },
            },
            "required": ["app_name"],
        },
    ),
    types.FunctionDeclaration(
        name="open_browser",
        description=(
            "Open the default web browser. Use when the user says 'open browser', "
            "'buka browser', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="open_url",
        description=(
            "Open a specific URL in the default browser. Use when the user asks "
            "to open a website like 'buka youtube', 'open github.com', etc."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "Full URL to open, e.g. 'https://youtube.com'.",
                },
            },
            "required": ["url"],
        },
    ),
    types.FunctionDeclaration(
        name="open_terminal",
        description=(
            "Open a terminal window. Use when the user says 'open terminal', "
            "'buka terminal', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="open_file_manager",
        description=(
            "Open the file manager (Explorer). Use when the user says 'open explorer', "
            "'buka file manager', 'buka explorer', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── System Power ─────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="lock_screen",
        description=(
            "Lock the computer screen. Use when the user says 'lock screen', "
            "'kunci layar', 'lock komputer', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="shutdown_pc",
        description=(
            "Schedule a system shutdown with a countdown. Default 60 seconds delay. "
            "Use when the user says 'shutdown', 'matikan komputer', 'turn off pc'. "
            "Always confirm with the user before executing."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "integer",
                    "description": "Seconds before shutdown (default 60).",
                },
            },
        },
    ),
    types.FunctionDeclaration(
        name="restart_pc",
        description=(
            "Schedule a system restart with a countdown. Default 60 seconds delay. "
            "Use when the user says 'restart', 'restart komputer', 'reboot'. "
            "Always confirm with the user before executing."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "delay_seconds": {
                    "type": "integer",
                    "description": "Seconds before restart (default 60).",
                },
            },
        },
    ),
    types.FunctionDeclaration(
        name="sleep_pc",
        description=(
            "Put the PC to sleep mode. Use when the user says 'sleep', "
            "'tidurkan komputer', 'sleep mode', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Screenshot ───────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="take_screenshot",
        description=(
            "Take a screenshot and save it. Use when the user says 'screenshot', "
            "'ambil screenshot', 'tangkap layar', 'take screenshot', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Timer ────────────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="set_timer",
        description=(
            "Set a countdown timer that shows a notification when done. "
            "Use when the user says 'set timer 5 menit', 'timer 30 detik', "
            "'remind me in 10 minutes', or similar. Convert to seconds."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "seconds": {
                    "type": "integer",
                    "description": "Timer duration in seconds.",
                },
                "label": {
                    "type": "string",
                    "description": "Description for the notification (default 'Timer').",
                },
            },
            "required": ["seconds"],
        },
    ),
    # ── Web Search ───────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="web_search",
        description=(
            "Search the web for current information. Use this when the user asks about "
            "current events, news, weather, recent facts, or anything you don't know or "
            "that may have changed after your training cutoff. "
            "Examples: 'siapa presiden Indonesia sekarang', 'cuaca Jakarta hari ini', "
            "'berita terbaru', 'what happened today', 'latest news'. "
            "Do NOT use this for time/date queries (use get_current_time instead) or "
            "for general knowledge you already know."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query to look up on the web.",
                },
            },
            "required": ["query"],
        },
    ),
    # ── User Memory ──────────────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="memory_store",
        description=(
            "Store a fact about the user in persistent memory. "
            "Use when the user tells you personal information like "
            "their name, location, preferences, hobbies, etc. "
            "Examples: 'nama saya Zhafran', 'I live in Bekasi'. "
            "Choose a short descriptive key and store the value."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": (
                        "Short identifier, e.g. 'name', 'location', "
                        "'hobby', 'favorite_food'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": "The fact value.",
                },
            },
            "required": ["key", "value"],
        },
    ),
    types.FunctionDeclaration(
        name="memory_search",
        description=(
            "Search stored memories by query. Use when the user "
            "asks if you remember something, asks 'siapa nama "
            "saya', 'kamu ingat saya?', 'what do you know about "
            "me', or to recall specific information."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "What to search for in memory.",
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="memory_forget",
        description=(
            "Remove a specific fact from memory. Use when the user "
            "says 'lupakan nama saya', 'forget my hobby', etc."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The fact key to forget.",
                },
            },
            "required": ["key"],
        },
    ),
    types.FunctionDeclaration(
        name="update_user_profile",
        description=(
            "Add information to the user's profile. Use for "
            "significant preferences or traits worth preserving."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "info": {
                    "type": "string",
                    "description": "Text to add to profile.",
                },
            },
            "required": ["info"],
        },
    ),
    # ── System Info ──────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="get_battery_level",
        description=(
            "Get battery percentage and charging status. Use when the user asks "
            "'berapa persen baterai', 'battery level', 'baterai tinggal berapa', "
            "'is it charging', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_ram_usage",
        description=(
            "Get current RAM/memory usage in GB and percentage. Use when the user asks "
            "'berapa RAM terpakai', 'memory usage', 'cek RAM', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_storage_info",
        description=(
            "Get disk storage usage (used/total/free in GB). Use when the user asks "
            "'berapa sisa storage', 'disk space', 'cek penyimpanan', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_ip_address",
        description=(
            "Get the local and public IP addresses. Use when the user asks "
            "'berapa IP saya', 'what is my IP', 'cek IP address', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_system_uptime",
        description=(
            "Get how long the system has been running since last boot. Use when the user "
            "asks 'sudah berapa lama menyala', 'uptime', 'kapan terakhir restart', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Quick Notes ──────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="add_note",
        description=(
            "Save a quick note with a timestamp. Use when the user says "
            "'catat', 'note', 'tulis catatan', 'save note', 'tambah catatan', or similar. "
            "Examples: 'catat: beli kopi besok', 'note: meeting jam 3'."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The note content to save.",
                },
            },
            "required": ["text"],
        },
    ),
    types.FunctionDeclaration(
        name="get_notes",
        description=(
            "Read the last 10 saved notes. Use when the user asks "
            "'lihat catatan', 'baca catatan', 'show notes', 'my notes', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="clear_notes",
        description=(
            "Delete all saved notes. Use when the user says "
            "'hapus semua catatan', 'clear notes', 'delete all notes', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Reminders (Heartbeat) ────────────────────────────────────────
    types.FunctionDeclaration(
        name="set_reminder",
        description=(
            "Set a reminder. For relative times ('2 menit lagi', 'setengah jam lagi', "
            "'1 jam lagi'), use delay_minutes. For absolute times ('besok jam 8', "
            "'jam 3 sore'), use remind_at with ISO 8601. One of remind_at or "
            "delay_minutes must be provided. If both given, delay_minutes wins. "
            "Jika user minta 'matiin AC 1 jam lagi', 'nyalain tv besok jam 8', dll., "
            "isi juga field 'action' agar perangkat dikontrol otomatis saat reminder fire."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "The reminder message.",
                },
                "remind_at": {
                    "type": "string",
                    "description": (
                        "ISO 8601 datetime for when to remind, "
                        "e.g. '2026-03-02T08:00:00'. "
                        "Use for absolute times like 'besok jam 8'."
                    ),
                },
                "delay_minutes": {
                    "type": "integer",
                    "description": (
                        "Minutes from now to remind. Use for relative times: "
                        "'2 menit lagi' → 2, 'setengah jam lagi' → 30, "
                        "'1 jam lagi' → 60. If provided, overrides remind_at."
                    ),
                },
                "lead_time": {
                    "type": "integer",
                    "description": "Minutes before remind_at to notify (default 5).",
                },
                "is_alarm": {
                    "type": "boolean",
                    "description": "If true, bypasses quiet hours (default false).",
                },
                "recurring": {
                    "type": "string",
                    "description": (
                        "Recurrence pattern: 'daily', 'weekly', 'weekdays', "
                        "or omit for one-time reminder."
                    ),
                },
                "action": {
                    "type": "object",
                    "description": (
                        "Opsional: IoT action yang otomatis dieksekusi saat reminder fire. "
                        "Gunakan saat user minta 'matiin AC 1 jam lagi', "
                        "'nyalain tv atas 20 menit lagi', 'matiin tv bawah besok', dll. "
                        "Jangan isi ini jika reminder hanya pengingat teks biasa."
                    ),
                    "properties": {
                        "device": {
                            "type": "string",
                            "description": "'ac', 'tv_atas', atau 'tv_bawah'.",
                        },
                        "command": {
                            "type": "string",
                            "description": (
                                "Perintah ke perangkat: "
                                "'on', 'off', 'set_temp', 'set_mode', 'set_fan', "
                                "'volume_up', 'volume_down', 'set_volume', dll."
                            ),
                        },
                        "value": {
                            "type": "string",
                            "description": (
                                "Opsional: nilai tambahan, misal suhu '24', "
                                "volume '50', nama app 'youtube'."
                            ),
                        },
                    },
                    "required": ["device", "command"],
                },
            },
            "required": ["message"],
        },
    ),
    types.FunctionDeclaration(
        name="list_reminders",
        description=(
            "List all pending (active) reminders. Use when the user asks "
            "'lihat reminder', 'reminder apa saja', 'show reminders', "
            "'ada reminder apa', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="cancel_reminder",
        description=(
            "Cancel a reminder by its ID number. Use when the user says "
            "'batalkan reminder 1', 'cancel reminder #2', 'hapus reminder', or similar. "
            "If the user doesn't specify an ID, list reminders first."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "reminder_id": {
                    "type": "integer",
                    "description": "The reminder ID to cancel.",
                },
            },
            "required": ["reminder_id"],
        },
    ),
    # ── Dictation ────────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="dictate",
        description=(
            "Type text into the currently active window (simulates keyboard input). "
            "Use when the user says 'ketik', 'type', 'tulis di layar', 'dictate', "
            "'ketikkan', or similar. The text will be typed into whatever app is focused. "
            "Examples: 'ketik: hello world', 'type this email for me'."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to type into the active window.",
                },
            },
            "required": ["text"],
        },
    ),
    # ── Display / Brightness ─────────────────────────────────────────
    types.FunctionDeclaration(
        name="brightness_up",
        description=(
            "Increase screen brightness by 10%. Use when the user says "
            "'naikkan brightness', 'brightness up', 'terangin layar', 'lebih terang', "
            "or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="brightness_down",
        description=(
            "Decrease screen brightness by 10%. Use when the user says "
            "'turunkan brightness', 'brightness down', 'redup layar', 'kurangi terang', "
            "or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_brightness",
        description=(
            "Get the current screen brightness level. Use when the user asks "
            "'berapa brightness', 'brightness level', 'tingkat kecerahan', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Network / Wi-Fi ──────────────────────────────────────────────
    types.FunctionDeclaration(
        name="wifi_on",
        description=(
            "Enable Wi-Fi. Use when the user says 'nyalakan wifi', 'wifi on', "
            "'aktifkan wifi', 'turn on wifi', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="wifi_off",
        description=(
            "Disable Wi-Fi. Use when the user says 'matikan wifi', 'wifi off', "
            "'nonaktifkan wifi', 'turn off wifi', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="get_wifi_status",
        description=(
            "Get current Wi-Fi connection status and connected SSID. Use when the user "
            "asks 'status wifi', 'wifi connected?', 'terhubung ke wifi apa', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Music Playback ───────────────────────────────────────────────
    types.FunctionDeclaration(
        name="play_music",
        description=(
            "Search and play a song. By default plays on YouTube Music in the browser. "
            "Can also play on a TV via WebOS YouTube app by setting target. "
            "Use when the user says 'puterin lagu', 'play song', 'putar musik', "
            "'play music', 'nyalakan lagu', 'mainkan lagu', or mentions a song/artist "
            "to play. If user says 'di TV Atas' or 'di TV', set target accordingly. "
            "Examples: 'puterin About You dari The 1975 di TV Atas', 'play Bohemian Rhapsody'. "
            "Build the query from song title and artist."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": (
                        "Song search query combining title and artist, "
                        "e.g. 'About You The 1975', 'Bohemian Rhapsody Queen'."
                    ),
                },
                "target": {
                    "type": "string",
                    "description": (
                        "Where to play: 'local' (laptop browser, default), "
                        "'tv_atas' (TV Atas via WebOS YouTube), "
                        "'tv_bawah' (TV Bawah via WebOS YouTube)."
                    ),
                    "enum": ["local", "tv_atas", "tv_bawah"],
                },
            },
            "required": ["query"],
        },
    ),
    types.FunctionDeclaration(
        name="pause_resume_music",
        description=(
            "Toggle play/pause on the currently playing music. Use when the user says "
            "'pause musik', 'resume musik', 'pause lagu', 'lanjutkan musik', "
            "'pause', 'resume', 'play', or similar while music is playing."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="skip_track",
        description=(
            "Skip to the next song/track. Use when the user says "
            "'skip', 'next song', 'lagu selanjutnya', 'ganti lagu', "
            "'next', 'skip lagu', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="previous_music_track",
        description=(
            "Go back to the previous song/track. Use when the user says "
            "'lagu sebelumnya', 'previous song', 'balik lagu', 'back', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    types.FunctionDeclaration(
        name="stop_music",
        description=(
            "Stop the currently playing music. Use when the user says "
            "'stop musik', 'hentikan musik', 'stop lagu', 'matikan musik', or similar."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {},
        },
    ),
    # ── Weather ─────────────────────────────────────────────────────
    types.FunctionDeclaration(
        name="get_weather",
        description=(
            "Ambil data prakiraan cuaca. Gunakan saat user bertanya tentang cuaca, "
            "suhu, hujan, atau kondisi luar. "
            "Contoh: 'cuaca hari ini', 'besok hujan nggak', 'weather in Tokyo', "
            "'prakiraan cuaca minggu ini'."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": (
                        "Nama kota. Kosongkan untuk Bekasi (lokasi default user)."
                    ),
                },
                "days": {
                    "type": "integer",
                    "description": "Jumlah hari prakiraan (1-7, default 3).",
                },
            },
        },
    ),
    # ── IoT / Smart Home ──────────────────────────────────────────────
    types.FunctionDeclaration(
        name="control_device",
        description=(
            "Kontrol perangkat smart home. Gunakan saat user ingin mengatur "
            "AC (nyala/mati/suhu/mode/kipas), TV (nyala/mati/volume/channel/"
            "buka app), atau perangkat IoT lainnya. "
            "Contoh: 'nyalakan AC', 'set suhu 24', 'matikan TV', "
            "'buka YouTube di TV atas', 'volume TV naik'."
        ),
        parameters_json_schema={
            "type": "object",
            "properties": {
                "device": {
                    "type": "string",
                    "description": (
                        "Nama perangkat: 'ac', 'tv_atas' (kamar atas), "
                        "atau 'tv_bawah' (ruang tamu)."
                    ),
                },
                "action": {
                    "type": "string",
                    "description": (
                        "Aksi: 'on', 'off', 'set_temp', 'set_mode', 'set_fan', "
                        "'volume_up', 'volume_down', 'set_volume', "
                        "'channel_up', 'channel_down', 'open_app', "
                        "'home', 'back', 'menu', 'up', 'down', 'left', 'right', 'ok'."
                    ),
                },
                "value": {
                    "type": "string",
                    "description": (
                        "Nilai opsional: angka suhu (16-30), level volume (0-100), "
                        "nama app ('youtube', 'netflix', 'disney', 'spotify'), "
                        "mode AC (0=dingin, 1=panas, 2=auto, 3=kipas, 4=kering), "
                        "atau kecepatan kipas (0=auto, 1=pelan, 2=sedang, 3=kencang)."
                    ),
                },
            },
            "required": ["device", "action"],
        },
    ),
]

# Map function names → async callables
_TOOL_IMPLEMENTATIONS: dict[str, object] = {
    # Time/Date
    "get_current_time": time_date.get_current_time,
    "get_current_date": time_date.get_current_date,
    "get_current_datetime": time_date.get_current_datetime,
    # Volume
    "volume_up": system_control.volume_up,
    "volume_down": system_control.volume_down,
    "mute_unmute": system_control.mute_unmute,
    # Media
    "play_pause_media": system_control.play_pause_media,
    "next_track": system_control.next_track,
    "previous_track": system_control.previous_track,
    # Apps
    "open_app": system_control.open_app,
    "open_browser": system_control.open_browser,
    "open_url": system_control.open_url,
    "open_terminal": system_control.open_terminal,
    "open_file_manager": system_control.open_file_manager,
    # Power
    "lock_screen": system_control.lock_screen,
    "shutdown_pc": system_control.shutdown_pc,
    "restart_pc": system_control.restart_pc,
    "sleep_pc": system_control.sleep_pc,
    # Screenshot & Timer
    "take_screenshot": system_control.take_screenshot,
    "set_timer": system_control.set_timer,
    # Web Search
    "web_search": web_search.web_search,
    # User Memory
    "memory_store": memory_store,
    "memory_search": memory_search,
    "memory_forget": memory_forget,
    "update_user_profile": update_user_profile,
    # Legacy aliases
    "remember_fact": remember_fact,
    "recall_facts": recall_facts,
    # System Info
    "get_battery_level": system_info.get_battery_level,
    "get_ram_usage": system_info.get_ram_usage,
    "get_storage_info": system_info.get_storage_info,
    "get_ip_address": system_info.get_ip_address,
    "get_system_uptime": system_info.get_system_uptime,
    # Quick Notes
    "add_note": notes.add_note,
    "get_notes": notes.get_notes,
    "clear_notes": notes.clear_notes,
    # Reminders (Heartbeat)
    "set_reminder": heartbeat_reminders.set_reminder,
    "list_reminders": heartbeat_reminders.list_reminders,
    "cancel_reminder": heartbeat_reminders.cancel_reminder,
    # Dictation
    "dictate": dictation.dictate,
    # Display / Brightness
    "brightness_up": display_control.brightness_up,
    "brightness_down": display_control.brightness_down,
    "get_brightness": display_control.get_brightness,
    # Network / Wi-Fi
    "wifi_on": network_control.wifi_on,
    "wifi_off": network_control.wifi_off,
    "get_wifi_status": network_control.get_wifi_status,
    # Music Playback
    "play_music": music_player.play_music,
    "pause_resume_music": music_player.pause_resume_music,
    "skip_track": music_player.skip_track,
    "previous_music_track": music_player.previous_music_track,
    "stop_music": music_player.stop_music,
    # Weather
    "get_weather": weather.get_weather,
    # IoT / Smart Home
    "control_device": iot.control_device,
}


def get_tool_declarations() -> list[types.Tool]:
    """Return the list of Tool objects for Gemini function calling.

    Returns:
        A list containing a single Tool with all function declarations.
    """
    return [types.Tool(function_declarations=_FUNCTION_DECLARATIONS)]


# Cached OpenAI-format tool list (built once on first call)
_openai_tools_cache: list[dict] | None = None


def get_tool_declarations_openai() -> list[dict]:
    """Return tool declarations in OpenAI function-calling format.

    Converts Gemini FunctionDeclaration objects to OpenAI-compatible dicts
    for use with Groq and other OpenAI-compatible APIs.

    Returns:
        List of tool dicts in OpenAI format.
    """
    global _openai_tools_cache
    if _openai_tools_cache is not None:
        return _openai_tools_cache

    tools = []
    for decl in _FUNCTION_DECLARATIONS:
        tools.append({
            "type": "function",
            "function": {
                "name": decl.name,
                "description": decl.description,
                "parameters": decl.parameters_json_schema or {"type": "object", "properties": {}},
            },
        })
    _openai_tools_cache = tools
    return tools


async def execute_tool(name: str, args: dict | None = None) -> str:
    """Execute a tool by name and return its result.

    Args:
        name: The function name as returned by the LLM function call.
        args: Arguments dict (most tools take none).

    Returns:
        The tool's result as a string.

    Raises:
        ValueError: If the tool name is unknown.
    """
    impl = _TOOL_IMPLEMENTATIONS.get(name)
    if impl is None:
        raise ValueError(f"Unknown tool: {name!r}")

    logger.info("Executing tool: %s(%s)", name, args or "")
    result = await impl(**(args or {}))
    logger.info("Tool %s result: %s", name, result)
    return result


def get_all_tool_names() -> list[str]:
    """Return all registered tool function names.

    Returns:
        Sorted list of tool name strings.
    """
    return sorted(_TOOL_IMPLEMENTATIONS.keys())
