# Menjalankan NOVA di Windows

Panduan ini untuk menjalankan NOVA **langsung di laptop/PC Windows** — semua tool (volume, buka app, screenshot, dll.) mengontrol mesin Windows itu sendiri.

> Kalau kamu mau jalankan NOVA di Ubuntu server dan mengontrol Windows dari jarak jauh, lihat [UBUNTU_SERVER.md](UBUNTU_SERVER.md) + [REMOTE_AGENT.md](REMOTE_AGENT.md).

---

## Prasyarat

| Kebutuhan | Minimum |
|-----------|---------|
| OS | Windows 10/11 |
| Python | 3.10+ |
| RAM | 2 GB free |
| Internet | Wajib (semua AI di cloud) |
| Mic + Speaker | Untuk voice mode |

---

## 1. Clone dan Install

```bash
git clone https://github.com/user/nova.git
cd nova

# Buat virtual environment (direkomendasikan)
python -m venv .venv
.venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
```

### Dependencies Tambahan (Opsional tapi Direkomendasikan)

```bash
# yt-dlp untuk fitur play music
pip install yt-dlp

# openwakeword untuk deteksi wake word "Hey NOVA"
pip install openwakeword
```

---

## 2. Konfigurasi API Keys

```bash
copy .env.example .env
```

Buka `.env` dan isi minimal satu API key LLM:

```dotenv
# Wajib (minimal salah satu)
NOVA_GEMINI_API_KEY=your_key_here     # https://ai.google.dev
NOVA_GROQ_API_KEY=your_key_here       # https://console.groq.com

# Opsional
NOVA_CLOUDFLARE_ACCOUNT_ID=...
NOVA_CLOUDFLARE_API_TOKEN=...
NOVA_GOOGLE_CLOUD_TTS_KEY_PATH=path/to/service-account.json
```

---

## 3. Verifikasi Setup

```bash
python -m nova --check
```

Output yang diharapkan:

```
NOVA System Check

  ✅ STT (groq): Available
  ✅ LLM (gemini): Available
  ✅ TTS (edge): Available
  ✅ wake_word: models/hey_nova.onnx found
  ✅ google_cloud_tts: connected (0 / 950,000 chars used)

All systems operational.
```

---

## 4. Menjalankan NOVA

### Mode Text (tanpa mikrofon)

```bash
python -m nova --text-only
```

### Mode Voice (push-to-talk)

```bash
python -m nova --push-to-talk
```

Tekan **Enter** untuk mulai berbicara, NOVA akan mendengarkan sampai kamu berhenti.

### Mode Wake Word (always listening)

```bash
python -m nova
```

NOVA akan mendengarkan terus-menerus dan aktif saat mendeteksi "Hey NOVA" atau double clap.

Kalau model wake word tidak tersedia, otomatis fallback ke **hotkey mode** (Ctrl+Space).

```bash
# Paksa hotkey mode
python -m nova --hotkey
```

---

## 5. Fitur yang Tersedia di Windows

Semua fitur berikut berjalan langsung di mesin Windows:

| Kategori | Fitur |
|----------|-------|
| **Volume** | Naikkan, turunkan, mute/unmute |
| **Media** | Play/pause, next track, previous track |
| **Aplikasi** | Buka Spotify, Notepad, Calculator, VS Code, dll. |
| **Browser** | Buka browser, buka URL spesifik |
| **Power** | Shutdown, restart, sleep, lock screen |
| **Screenshot** | Ambil screenshot, simpan ke ~/Pictures/Screenshots/ |
| **Timer** | Set countdown timer dengan notifikasi |
| **Musik** | Cari dan putar lagu di YouTube Music |
| **Dictation** | Ketik teks ke window aktif (voice-to-type) |
| **Brightness** | Atur kecerahan layar naik/turun |
| **Wi-Fi** | Nyalakan/matikan Wi-Fi, cek status |
| **System Info** | Cek baterai, RAM, storage, IP, uptime |
| **Web Search** | Cari di DuckDuckGo |
| **Weather** | Prakiraan cuaca (Open-Meteo) |
| **Notes** | Catat, baca, hapus catatan |
| **Reminders** | Set reminder dengan waktu spesifik |
| **Memory** | NOVA mengingat fakta tentang kamu |
| **IoT** | Kontrol AC (Tuya), TV (LG WebOS) via jaringan |

---

## 6. Konfigurasi Tambahan

### Menonaktifkan Remote Agent Server

Secara default, NOVA membuka WebSocket server di port 8765 untuk menerima koneksi agent. Kalau tidak butuh fitur remote agent:

```dotenv
# .env
NOVA_REMOTE_AGENT_ENABLED=false
```

### Menonaktifkan Clap Detection

```dotenv
NOVA_CLAP_DETECTION_ENABLED=false
```

### Mengubah Port Remote Agent

```dotenv
NOVA_REMOTE_AGENT_PORT=9000
```

---

## 7. Cek Quota Google TTS

```bash
python -m nova --quota
```

---

## Langkah Selanjutnya

- Untuk deploy di Ubuntu server: [UBUNTU_SERVER.md](UBUNTU_SERVER.md)
- Untuk kontrol Windows dari jarak jauh: [REMOTE_AGENT.md](REMOTE_AGENT.md)
- Masalah? Lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
