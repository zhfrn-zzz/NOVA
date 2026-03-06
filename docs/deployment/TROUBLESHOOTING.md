# Troubleshooting NOVA

Daftar masalah umum beserta solusinya.

---

## Daftar Isi

- [Audio & Mikrofon](#audio--mikrofon)
- [API & Provider](#api--provider)
- [Wake Word](#wake-word)
- [Remote Agent (WebSocket)](#remote-agent-websocket)
- [TTS (Text-to-Speech)](#tts-text-to-speech)
- [Tool Execution](#tool-execution)
- [Performa & RAM](#performa--ram)
- [IoT (AC / TV)](#iot-ac--tv)
- [Systemd Service (Ubuntu)](#systemd-service-ubuntu)

---

## Audio & Mikrofon

### Mic tidak terdeteksi

**Gejala:** `sounddevice.PortAudioError` atau NOVA tidak merespon suara.

**Ubuntu:**

```bash
# List audio devices
python3 -c "import sounddevice; print(sounddevice.query_devices())"

# Test rekam
arecord -d 3 test.wav && aplay test.wav

# Jika mic USB tidak jadi default
echo 'defaults.pcm.card 1' >> ~/.asoundrc
```

**Windows:**

1. Buka **Settings → Sound → Input** — pastikan mic yang benar dipilih
2. Test di **Voice Recorder** app
3. Pastikan tidak di-mute

### mpv tidak ditemukan (TTS tidak keluar suara)

**Ubuntu:**

```bash
sudo apt install mpv
```

**Windows:**

Download mpv dari https://mpv.io/installation/ dan pastikan ada di PATH, atau install via Chocolatey:

```bash
choco install mpv
```

### Audio delay / playback lambat

Pastikan `mpv` menggunakan PulseAudio (Ubuntu):

```bash
mpv --ao=pulse test.wav
```

---

## API & Provider

### `401 Unauthorized` / `403 Forbidden`

API key salah atau expired.

```bash
# Cek .env sudah benar
cat .env | grep NOVA_

# Test manual
curl -H "Authorization: Bearer YOUR_KEY" https://generativelanguage.googleapis.com/...
```

**Solusi:** Generate ulang API key di dashboard provider masing-masing.

### `429 Too Many Requests` (rate limit)

Free tier punya batas request. NOVA otomatis failover ke provider berikutnya.

| Provider | Limit |
|----------|-------|
| Gemini | 15 RPM, 1500 req/day |
| Groq | 30 RPM |
| Cloudflare | 10K neurons/day |

**Solusi:** Tunggu beberapa menit, atau tambah lebih banyak provider di `.env`.

### `Connection Error` / timeout

Internet mati atau provider down.

```bash
# Cek koneksi
ping google.com

# Cek status provider
curl -s https://status.groq.com
```

### Semua provider gagal

```
ERROR: All STT providers failed
```

**Cek:**
1. Internet aktif?
2. Minimal satu API key diisi di `.env`?
3. Jalankan `python -m nova --check` untuk verifikasi

---

## Wake Word

### "Hey NOVA" tidak terdeteksi

1. **Model tidak ditemukan:**

   ```
   WARNING: Wake word model not found: models/hey_nova.onnx
   ```

   Solusi: copy model file ke `models/hey_nova.onnx`

2. **openwakeword tidak terinstall:**

   ```bash
   pip install openwakeword
   ```

3. **Mic sensitivity terlalu rendah:**
   - Bicara lebih dekat ke mic
   - Cek input level di sound settings

4. **Fallback ke hotkey mode:**
   Jika openwakeword gagal, NOVA otomatis gunakan `Ctrl+Space`. Lihat log:

   ```
   INFO: openwakeword unavailable, using keyboard hotkey fallback
   ```

### Double clap tidak terdeteksi

```dotenv
# Pastikan di .env
NOVA_CLAP_DETECTION_ENABLED=true
```

Jika terlalu sensitif atau kurang sensitif, sesuaikan threshold di `audio/clap_detector.py`.

---

## Remote Agent (WebSocket)

### Agent tidak bisa connect ke server

**Gejala:** `ConnectionRefusedError` atau `TimeoutError`

**Cek di sisi server (Ubuntu):**

```bash
# NOVA berjalan?
sudo systemctl status nova

# Port terbuka?
ss -tlnp | grep 8765

# Firewall?
sudo ufw status
sudo ufw allow 8765/tcp
```

**Cek di sisi agent (Windows):**

```bash
# Server URL benar?
# Format: ws://IP_SERVER:8765 (bukan http://)

# Test koneksi
python -c "import asyncio, websockets; asyncio.run(websockets.connect('ws://192.168.1.100:8765'))"
```

**Penyebab umum:**

| Masalah | Solusi |
|---------|--------|
| IP salah | Jalankan `hostname -I` di server |
| Port salah | Cek `NOVA_REMOTE_AGENT_PORT` di `.env` server |
| Firewall blok | `sudo ufw allow 8765/tcp` |
| Beda network | Pastikan kedua device di Wi-Fi/LAN yang sama |
| Server belum jalan | Start NOVA dulu di server |

### Auth gagal

```
ERROR: Authentication failed
```

**Solusi:** Pastikan token sama di kedua sisi:

- Server (`.env`): `NOVA_REMOTE_AGENT_TOKEN=rahasia123`
- Agent (config.json atau CLI): `--token rahasia123`

Atau kosongkan keduanya untuk disable auth.

### Tool tidak di-route ke remote agent

**Gejala:** Perintah "buka Spotify" mengeksekusi di server, bukan di Windows.

**Cek:**

1. Agent terconnect?
   ```
   INFO: Remote agent registered: windows-laptop
   ```

2. Tool masuk daftar remote? (lihat `REMOTE_TOOLS` di `src/nova/remote/server.py`)

3. Log routing:
   ```
   INFO: Routing tool to remote agent (windows-laptop): open_app({'name': 'spotify'})
   ```

### Agent sering disconnect

Bisa karena Wi-Fi tidak stabil. Agent otomatis reconnect setiap 5 detik. Kalau terlalu sering:

- Pindah ke koneksi ethernet (kabel LAN)
- Cek kualitas Wi-Fi: `ping 192.168.1.100 -t`

---

## TTS (Text-to-Speech)

### Google Cloud TTS error

```
ERROR: Google Cloud TTS failed: 403 ...
```

**Cek:**

1. Service account JSON ada dan path benar di `.env`:
   ```dotenv
   NOVA_GOOGLE_CLOUD_TTS_KEY_PATH=/path/to/service-account.json
   ```

2. Cloud Text-to-Speech API enabled di Google Cloud Console

3. Quota tersisa:
   ```bash
   python -m nova --quota
   ```

**Jika quota habis:** NOVA otomatis fallback ke Edge TTS (gratis, unlimited).

### Suara TTS tidak keluar

1. Cek `mpv` terinstall
2. Cek volume tidak mute
3. Coba test manual:
   ```bash
   edge-tts --text "halo dunia" --write-media test.mp3 && mpv test.mp3
   ```

### TTS terlalu lambat (high latency)

NOVA menggunakan streaming TTS (split per kalimat). Jika masih lambat:

- Internet lambat? Cek speed
- Pakai Edge TTS sebagai primary (lebih cepat, tidak perlu auth):
  ```dotenv
  # Komentari Google Cloud TTS key untuk force Edge TTS
  # NOVA_GOOGLE_CLOUD_TTS_KEY_PATH=
  ```

---

## Tool Execution

### "Maaf, saya tidak bisa melakukan itu"

NOVA tidak mengenali intent sebagai tool call. Coba ulang dengan perintah lebih eksplisit:

- "Naikkan volume" (bukan "tolong suaranya kurang keras")
- "Buka Chrome" (bukan "aku mau browsing")
- "Berapa baterai?" (bukan "apakah laptopku masih hidup lama?")

### open_app gagal di Windows

```
ERROR: App 'spotify' not found
```

NOVA mencari app di Start Menu. Pastikan nama cocok:

| Perintah | Yang dicari |
|----------|-------------|
| "Buka Spotify" | `spotify.exe` atau Start Menu shortcut "Spotify" |
| "Buka VS Code" | `code` atau "Visual Studio Code" |
| "Buka Calculator" | `calc.exe` |

### Screenshot gagal

**Windows:** Butuh `pyautogui` — sudah ada di requirements.

**Ubuntu:** Butuh desktop environment:

```bash
sudo apt install gnome-screenshot
```

---

## Performa & RAM

### NOVA pakai RAM > 300 MB

```bash
# Monitor
ps aux | grep nova
htop
```

**Penyebab umum:**

| Penyebab | Solusi |
|----------|--------|
| Embedding enabled | `NOVA_EMBEDDING_ENABLED=false` |
| Memory store besar | NOVA auto-compact, tapi bisa clear manual |
| TTS cache penuh | Restart NOVA (cache in-memory, auto-clear) |

### CPU usage tinggi

Biasanya dari:
- openwakeword (selalu mendengarkan) → nonaktifkan jika tidak dipakai
- Clap detection → `NOVA_CLAP_DETECTION_ENABLED=false`

---

## IoT (AC / TV)

### Tuya (AC) error

```
ERROR: Tuya API error: sign invalid
```

**Cek:**
1. `TUYA_ACCESS_ID` dan `TUYA_ACCESS_KEY` benar
2. Region benar (`us`, `eu`, `cn`, `in`)
3. Device terdaftar di iot.tuya.com dan linked ke project

### TV WebOS tidak terhubung

```
ERROR: Cannot connect to TV at 192.168.x.x
```

**Cek:**
1. TV menyala
2. IP TV benar (cek di Settings → Network → Connection info)
3. Di jaringan yang sama
4. Pertama kali: perlu pairing
   ```bash
   python scripts/pair_tv.py
   ```
   Terima prompt "Allow" di TV

---

## Systemd Service (Ubuntu)

### Service gagal start

```bash
sudo systemctl status nova
# Lihat error

journalctl -u nova -n 50 --no-pager
# Lihat log lengkap
```

**Penyebab umum:**

| Error | Solusi |
|-------|--------|
| `ModuleNotFoundError` | Path venv salah di `ExecStart` |
| `PermissionError: /dev/snd` | Tambah user ke group `audio`: `sudo usermod -aG audio $USER` |
| `DISPLAY not set` | Tambah `Environment=DISPLAY=:0` di service file |
| `.env not found` | Pastikan `WorkingDirectory` benar |

### Service tidak bisa akses audio

```bash
# Tambah user ke audio group
sudo usermod -aG audio your_username

# Logout/login ulang, lalu restart service
sudo systemctl restart nova
```

### Lihat log real-time

```bash
journalctl -u nova -f
```

---

## Masih Bermasalah?

1. **Cek log:** `journalctl -u nova -f` (Ubuntu) atau `~/.nova-agent/agent.log` (Windows agent)
2. **Mode verbose:** Tambah `NOVA_LOG_LEVEL=DEBUG` di `.env`
3. **Test individual:** Jalankan `python -m nova --text-only` dulu sebelum voice mode
4. **System check:** `python -m nova --check`
