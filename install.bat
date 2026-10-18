@echo off
cd /d "%~dp0"
echo === Lab Monitor install ===

where python >nul 2>nul
if errorlevel 1 (
    echo Python not found. Install Python 3.9+ from https://www.python.org
    echo and tick "Add python.exe to PATH" during setup, then run this again.
    pause
    exit /b 1
)

if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 goto fail
)

echo Installing packages...
venv\Scripts\python.exe -m pip install --upgrade pip
venv\Scripts\python.exe -m pip install "opencv-python>=4.8,<5" numpy flask pyserial
if errorlevel 1 goto fail

echo Downloading face models...
venv\Scripts\python.exe monitor.py --download
if errorlevel 1 goto fail

venv\Scripts\python.exe monitor.py --selftest
if errorlevel 1 goto fail

echo.
echo Install complete. Start the app with run.bat
pause
exit /b 0

:fail
echo.
echo Install FAILED - see the error above.
pause
exit /b 1
