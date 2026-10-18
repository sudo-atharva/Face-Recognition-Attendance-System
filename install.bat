@echo off
setlocal
cd /d "%~dp0"
echo === Lab Monitor install ===
echo Installs Python if missing, the Python packages, the face models and a desktop shortcut.
echo.

call :findpy
if not defined PY (
    echo Python 3.9+ not found - installing Python 3.12 for this user with winget...
    winget install -e --id Python.Python.3.12 --scope user --silent --accept-package-agreements --accept-source-agreements
    call :findpy
)
if not defined PY (
    echo winget not available or failed - downloading Python from python.org...
    curl -L --fail -o "%TEMP%\python-setup.exe" https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe || goto fail
    "%TEMP%\python-setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 || goto fail
    del "%TEMP%\python-setup.exe"
    call :findpy
)
if not defined PY (
    echo Could not install Python. Install Python 3.9+ from https://www.python.org
    echo and tick "Add python.exe to PATH", then run this again.
    goto fail
)
echo Using Python: %PY%

rem A venv breaks if the Python it was made from is removed: rebuild it
if exist venv\Scripts\python.exe venv\Scripts\python.exe -c "import sys" >nul 2>nul || rmdir /s /q venv
if not exist venv\Scripts\python.exe (
    echo Creating virtual environment...
    %PY% -m venv venv || goto fail
)

echo Installing packages...
venv\Scripts\python.exe -m pip install --upgrade pip || goto fail
venv\Scripts\python.exe -m pip install "opencv-python>=4.8,<5" numpy flask pyserial || goto fail

echo Downloading face models...
venv\Scripts\python.exe monitor.py --download || goto fail
venv\Scripts\python.exe monitor.py --selftest || goto fail

echo Creating desktop shortcut "Lab Monitor"...
powershell -NoProfile -Command "$s = (New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Desktop') + '\Lab Monitor.lnk'); $s.TargetPath = '%~dp0run.bat'; $s.WorkingDirectory = '%~dp0'; $s.Save()"

echo.
echo Install complete. Start the app with the "Lab Monitor" desktop shortcut or run.bat
pause
exit /b 0

:fail
echo.
echo Install FAILED - see the error above.
pause
exit /b 1

rem Sets PY to a working Python 3.9+ command. Skips the Microsoft Store "python" stub.
:findpy
set "PY="
py -3 -c "import sys; assert sys.version_info >= (3, 9)" >nul 2>nul && set "PY=py -3" && exit /b
python -c "import sys; assert sys.version_info >= (3, 9)" >nul 2>nul && set "PY=python" && exit /b
rem fresh install: PATH of this window is not updated yet
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set PY="%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
exit /b
