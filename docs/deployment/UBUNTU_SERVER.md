# Menjalankan NOVA di Ubuntu Server

Panduan ini untuk menjalankan NOVA di **Ubuntu server** (misalnya ASUS E410MA) sebagai voice assistant yang selalu aktif — seperti smart speaker.

> Untuk mengontrol laptop Windows dari jarak jauh, kombinasikan panduan ini dengan [REMOTE_AGENT.md](REMOTE_AGENT.md).

---

## Arsitektur

```
┌──────────────────────────────────┐
│  Ubuntu Server (NOVA)            │
│  - Mic + Speaker terpasang       │
│  - Selalu menyala                │
│  - Pipeline: Mic→STT→LLM→TTS    │
│  - Kontrol IoT (AC, TV)          │
│  - WebSocket server (:8765)      │
└──────────┬───────────────────────┘
           │ WebSocket (LAN)
           ▼
┌──────────────────────────────────┐
│  Windows Laptop (Agent)          │  ← Opsional
│  - Kontrol sistem (volume, app)  │
│  - Auto-connect saat nyala       │
└──────────────────────────────────┘
```

---

## Prasyarat

| Kebutuhan | Spec |
|-----------|------|
| OS | Ubuntu 22.04+ LTS |
| Python | 3.10+ |
| RAM | 4 GB (NOVA butuh < 300 MB) |
| Audio | Mic USB + Speaker/headphone |
| Internet | Wajib |

---

## 1. Install System Dependencies

```bash
sudo apt update
sudo apt install -y \
    python3 python3-pip python3-venv \
    portaudio19-dev \
    mpv \
    ffmpeg \
    libnotify-bin \
    git
```

### Penjelasan Packages

| Package | Fungsi |
|---------|--------|
| `portaudio19-dev` | Audio capture (sounddevice) |
| `mpv` | Audio playback (TTS) |
| `ffmpeg` | Audio conversion |
| `libnotify-bin` | Desktop notification (`notify-send`) untuk timer |

---

## 2. Clone dan Install NOVA

```bash
git clone https://github.com/user/nova.git
cd nova

python3 -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
```

### Dependencies Opsional

```bash
# Wake word detection
pip install openwakeword

# Music search
pip install yt-dlp
```

---

## 3. Konfigurasi

```bash
cp .env.example .env
nano .env
```

Isi minimal:

```dotenv
NOVA_GEMINI_API_KEY=your_key_here
NOVA_GROQ_API_KEY=your_key_here
```

### Konfigurasi Remote Agent (opsional)

Tambahkan jika ingin mengontrol Windows laptop dari jarak jauh:

```dotenv
# Remote agent
NOVA_REMOTE_AGENT_ENABLED=true
NOVA_REMOTE_AGENT_PORT=8765
NOVA_REMOTE_AGENT_TOKEN=rahasia123  # Kosongkan jika tidak perlu auth
```

---

## 4. Setup Audio

### Cek Mic

```bash
# List audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Test recording (5 detik)
arecord -d 5 test.wav && aplay test.wav
```

### Cek Speaker

```bash
# Test playback
mpv /usr/share/sounds/freedesktop/stereo/complete.oga
```

Jika mic USB tidak terdeteksi sebagai default:

```bash
# Set default input device
echo 'defaults.pcm.card 1' >> ~/.asoundrc
```

---

## 5. Setup Wake Word Model

Salin model wake word `hey_nova.onnx` ke folder models:

```bash
mkdir -p models
# Copy model file
cp /path/to/hey_nova.onnx models/
```

Verifikasi:

```bash
python3 -m nova --check
```

---

## 6. Menjalankan NOVA

### Manual (foreground)

```bash
source .venv/bin/activate

# Wake word mode (default — selalu mendengarkan)
python3 -m nova

# Atau text-only untuk testing
python3 -m nova --text-only
```

### Sebagai Systemd Service (auto-start)

Buat file service:

```bash
sudo nano /etc/systemd/system/nova.service
```

```ini
[Unit]
Description=NOVA Voice Assistant
After=network-online.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/home/your_username/nova
ExecStart=/home/your_username/nova/.venv/bin/python -m nova
Restart=on-failure
RestartSec=5
Environment=DISPLAY=:0
Environment=PULSE_SERVER=unix:/run/user/1000/pulse/native

[Install]
WantedBy=multi-user.target
```

> Ganti `your_username` dengan username kamu.

Aktifkan:

```bash
sudo systemctl daemon-reload
sudo systemctl enable nova
sudo systemctl start nova

# Cek status
sudo systemctl status nova

# Lihat log
journalctl -u nova -f
```

---

## 7. Fitur yang Tersedia di Ubuntu

### Berjalan Penuh (tanpa remote agent)

| Kategori | Status |
|----------|--------|
| Voice pipeline (STT → LLM → TTS) | Jalan |
| IoT (AC, TV) | Jalan (via jaringan) |
| Weather, Web Search | Jalan (cloud API) |
| Notes, Reminders, Memory | Jalan (lokal SQLite) |
| Time/Date | Jalan |
| Wake Word / Clap Detection | Jalan |

### Terbatas Tanpa Remote Agent

| Kategori | Status | Catatan |
|----------|--------|---------|
| Volume/Media | Terbatas | Tidak ada PulseAudio SendKeys equivalent |
| Open App | Terbatas | Butuh `xdg-open` atau app name Linux |
| Screenshot | Butuh GNOME | `gnome-screenshot` |
| Brightness | Butuh setup | `screen-brightness-control` + udev |
| Wi-Fi | Jalan | Via `nmcli` |
| Dictation | Butuh X11 | `pyautogui` tidak jalan di headless |

### Dengan Remote Agent di Windows

Semua tool di atas **berjalan penuh** — di-route ke laptop Windows via WebSocket. Lihat [REMOTE_AGENT.md](REMOTE_AGENT.md).

---

## 8. Firewall

Jika menggunakan remote agent, buka port WebSocket:

```bash
sudo ufw allow 8765/tcp comment "NOVA Remote Agent"
```

---

## 9. Tips Performa (Low-Spec Hardware)

### Monitor RAM

```bash
# Cek penggunaan NOVA
ps aux | grep nova | grep -v grep
htop  # Interaktif
```

NOVA harus di bawah **300 MB RAM**. Kalau lebih, ada yang salah.

### Optimasi

```dotenv
# .env — nonaktifkan fitur yang tidak dipakai
NOVA_EMBEDDING_ENABLED=false        # Hemat RAM ~50MB
NOVA_CLAP_DETECTION_ENABLED=false   # Hemat CPU
NOVA_CUSTOM_SOUNDS_ENABLED=false    # Hemat RAM
```

---

## Langkah Selanjutnya

- Setup remote agent di Windows: [REMOTE_AGENT.md](REMOTE_AGENT.md)
- Masalah? Lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
