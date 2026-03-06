@echo off
REM ── Build NOVA Agent as a standalone .exe ───────────────────────────
REM Requires: pip install pyinstaller
REM Output:   dist\NOVA Agent.exe

echo Installing build dependencies...
pip install pyinstaller -q

echo.
echo Building NOVA Agent...
pyinstaller ^
    --onefile ^
    --noconsole ^
    --name "NOVA Agent" ^
    --add-data "*.json;." ^
    --hidden-import pystray._win32 ^
    nova_agent.py

echo.
if exist "dist\NOVA Agent.exe" (
    echo Build successful!
    echo.
    echo Output: dist\NOVA Agent.exe
    echo Size:
    for %%A in ("dist\NOVA Agent.exe") do echo   %%~zA bytes
    echo.
    echo You can now copy "dist\NOVA Agent.exe" to any Windows PC.
    echo No Python installation required on the target machine.
) else (
    echo Build failed. Check the output above for errors.
)

echo.
pause
