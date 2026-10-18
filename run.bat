@echo off
cd /d "%~dp0"
if not exist venv\Scripts\pythonw.exe (
    echo Not installed yet - running install.bat first...
    call install.bat
    if not exist venv\Scripts\pythonw.exe exit /b 1
)
rem Starts the website in the background and opens it in the browser.
rem Log: server.log. To stop: close it from Task Manager (pythonw.exe) or run stop.bat
start "" venv\Scripts\pythonw.exe app.py --open
