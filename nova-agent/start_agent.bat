@echo off
REM ── NOVA Remote Agent (Console Mode) ────────────────────────────────
REM Runs in console mode for debugging. Use the .exe or pythonw for
REM background operation with system tray.

cd /d "%~dp0"
python nova_agent.py --console --server ws://192.168.1.100:8765
pause
