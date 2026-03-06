@echo off
REM ── Install NOVA Agent to Windows Startup ───────────────────────────
REM Works with both the .exe build and the Python source.
REM Creates a shortcut in the Startup folder for auto-start on login.

echo.
echo Installing NOVA Agent to Windows Startup...
echo.

set STARTUP_DIR=%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup
set SHORTCUT_PATH=%STARTUP_DIR%\NOVA Agent.lnk

REM Detect whether to use the .exe or the Python script
if exist "%~dp0dist\NOVA Agent.exe" (
    set TARGET_PATH=%~dp0dist\NOVA Agent.exe
    set DISPLAY_TARGET=dist\NOVA Agent.exe
) else if exist "%~dp0nova_agent.py" (
    set TARGET_PATH=pythonw
    set TARGET_ARGS="%~dp0nova_agent.py"
    set DISPLAY_TARGET=nova_agent.py (via pythonw)
) else (
    echo ERROR: Could not find NOVA Agent.exe or nova_agent.py
    pause
    exit /b 1
)

REM Create shortcut via PowerShell
if defined TARGET_ARGS (
    powershell -Command "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut('%SHORTCUT_PATH%'); $sc.TargetPath = '%TARGET_PATH%'; $sc.Arguments = '%TARGET_ARGS%'; $sc.WorkingDirectory = '%~dp0'; $sc.Description = 'NOVA Remote Agent'; $sc.Save()"
) else (
    powershell -Command "$ws = New-Object -ComObject WScript.Shell; $sc = $ws.CreateShortcut('%SHORTCUT_PATH%'); $sc.TargetPath = '%TARGET_PATH%'; $sc.WorkingDirectory = '%~dp0'; $sc.Description = 'NOVA Remote Agent'; $sc.Save()"
)

if %ERRORLEVEL% EQU 0 (
    echo Done! NOVA Agent will start automatically on login.
    echo.
    echo   Target:   %DISPLAY_TARGET%
    echo   Shortcut: %SHORTCUT_PATH%
    echo.
    echo On first run, a Settings dialog will appear to configure the server URL.
) else (
    echo Failed to create startup shortcut.
)

echo.
pause
