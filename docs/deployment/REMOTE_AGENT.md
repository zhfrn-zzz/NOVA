# NOVA Remote Agent — Kontrol Windows dari Jarak Jauh

Panduan ini untuk menjalankan **NOVA Agent** di laptop Windows agar bisa dikontrol oleh NOVA server (Ubuntu) melalui jaringan lokal.

> Prasyarat: NOVA server sudah berjalan di Ubuntu. Lihat [UBUNTU_SERVER.md](UBUNTU_SERVER.md).

---

## Cara Kerja

```
Kamu bicara ke Ubuntu Server (smart speaker)
        ↓
  NOVA: "Buka Spotify"
        ↓ function_call: open_app("spotify")
        ↓
  Registry cek: ada remote agent? → Ya!
        ↓ WebSocket (LAN)
        ↓
  Windows Agent terima → jalankan subprocess
        ↓
  Spotify terbuka di laptop Windows
        ↓
  Agent kirim result → Server → TTS
        ↓
  NOVA: "Spotify telah dibuka."
```

---

## Opsi Deployment

| Opsi | Kelebihan | Untuk Siapa |
|------|-----------|-------------|
| **A. Dari source (Python)** | Mudah di-debug, bisa dimodifikasi | Developer |
| **B. Build ke .exe** | Standalone, tidak perlu Python | End user |

---

## Opsi A: Jalankan dari Source

### 1. Install Dependencies

```bash
cd nova-agent
pip install -r requirements.txt

# Opsional: untuk fitur play music
pip install yt-dlp
```

### 2. Pertama Kali — Mode GUI

```bash
python nova_agent.py
```

Saat pertama dijalankan, dialog **Settings** otomatis muncul:

1. Isi **Server URL**: `ws://192.168.x.x:8765` (ganti dengan IP Ubuntu server)
2. Isi **Auth Token**: sama dengan `NOVA_REMOTE_AGENT_TOKEN` di server (kosongkan jika tidak diset)
3. Isi **Device Name**: nama device (default: `windows-laptop`)
4. Klik **Save**

Config tersimpan di `~/.nova-agent/config.json`. Tidak perlu isi ulang.

### 3. Mode Console (untuk debugging)

```bash
python nova_agent.py --console --server ws://192.168.1.100:8765
```

Output:

```
2026-03-06 18:00:00 [INFO] NOVA Remote Agent starting...
2026-03-06 18:00:00 [INFO] Connecting to ws://192.168.1.100:8765 ...
2026-03-06 18:00:01 [INFO] Connected to ws://192.168.1.100:8765 as 'windows-laptop'
```

---

## Opsi B: Build ke .exe (Standalone)

### 1. Install Build Dependencies

```bash
cd nova-agent
pip install -r requirements.txt
pip install pyinstaller
```

### 2. Build

```bash
build.bat
```

Atau manual:

```bash
pyinstaller --onefile --noconsole --name "NOVA Agent" --hidden-import pystray._win32 nova_agent.py
```

Output: `dist\NOVA Agent.exe` (~15-25 MB)

### 3. Distribusi

Copy `NOVA Agent.exe` ke laptop Windows manapun. **Tidak perlu install Python.** Double-click untuk menjalankan.

---

## System Tray

Setelah berjalan, NOVA Agent muncul di **system tray** (area notifikasi, pojok kanan bawah):

### Ikon Status

| Ikon | Arti |
|------|------|
| Bulat **hijau** dengan "N" | Terhubung ke NOVA server |
| Bulat **merah** dengan "N" | Terputus |
| Bulat **kuning** dengan "N" | Sedang menghubungkan |

### Menu (klik kanan ikon)

| Menu | Fungsi |
|------|--------|
| **Status** | Tampilkan status koneksi |
| **Settings...** | Buka dialog konfigurasi |
| **View Logs** | Buka file log di Notepad |
| **Reconnect** | Paksa reconnect ke server |
| **Exit** | Keluar dari agent |

### Notifikasi

- Balloon notification muncul saat **connect** dan **disconnect**.

---

## Auto-Start saat Windows Login

### Cara 1: Via install script

```bash
install_startup.bat
```

Script ini otomatis detect apakah menggunakan `.exe` atau `.py`, lalu membuat shortcut di:

```
%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\NOVA Agent.lnk
```

### Cara 2: Manual

1. Tekan `Win + R`, ketik `shell:startup`, Enter
2. Copy shortcut `NOVA Agent.exe` ke folder yang terbuka
3. Atau: buat shortcut baru → target: `"C:\path\to\NOVA Agent.exe"`

### Cara 3: Task Scheduler (lebih reliable)

1. Buka **Task Scheduler** (`taskschd.msc`)
2. Create Basic Task → nama: "NOVA Agent"
3. Trigger: **When I log on**
4. Action: **Start a program** → browse ke `NOVA Agent.exe`
5. Finish

---

## Konfigurasi

### File Config

Config disimpan di: `%USERPROFILE%\.nova-agent\config.json`

```json
{
  "server_url": "ws://192.168.1.100:8765",
  "token": "",
  "device_name": "windows-laptop"
}
```

Cara edit:
- Via menu tray **Settings...**
- Atau edit file JSON langsung

### CLI Arguments (override config)

```bash
python nova_agent.py --server ws://192.168.1.100:8765 --token rahasia123 --device laptop-kantor
```

| Argument | Fungsi |
|----------|--------|
| `--server`, `-s` | WebSocket URL server |
| `--token`, `-t` | Auth token |
| `--device`, `-d` | Nama device |
| `--console` | Mode console tanpa GUI |

---

## Cara Mengetahui IP Ubuntu Server

Di Ubuntu server, jalankan:

```bash
hostname -I
# atau
ip addr show | grep "inet " | grep -v 127.0.0.1
```

Contoh output: `192.168.1.100`

Maka agent URL: `ws://192.168.1.100:8765`

---

## Verifikasi Koneksi

### Di sisi Ubuntu server

Saat agent terhubung, log NOVA menampilkan:

```
[INFO] New WebSocket connection from ('192.168.1.50', 54321)
[INFO] Remote agent registered: windows-laptop from ('192.168.1.50', 54321)
```

### Di sisi Windows agent

```
[INFO] Connected to ws://192.168.1.100:8765 as 'windows-laptop'
```

### Test

Bicara ke NOVA (di Ubuntu server):

> "Nova, buka Spotify"

Spotify harus terbuka di **laptop Windows**, bukan di server Ubuntu.

---

## Log

Log agent tersimpan di:

```
%USERPROFILE%\.nova-agent\agent.log
```

Buka via tray menu **View Logs** atau manual:

```bash
notepad %USERPROFILE%\.nova-agent\agent.log
```

---

## Selanjutnya

- Masalah? Lihat [TROUBLESHOOTING.md](TROUBLESHOOTING.md)
