@echo off
cd /d "%~dp0"
if not exist venv\Scripts\pythonw.exe (
    echo Not installed yet - running install.bat first...
    call install.bat
    if not exist venv\Scripts\pythonw.exe exit /b 1
)
start "" venv\Scripts\pythonw.exe lab_monitor.py
