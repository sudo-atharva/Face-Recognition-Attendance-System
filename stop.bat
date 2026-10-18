@echo off
rem Stops the Lab Monitor website started by run.bat
powershell -NoProfile -Command "Get-CimInstance Win32_Process -Filter \"Name='pythonw.exe'\" | Where-Object CommandLine -like '*app.py*' | Invoke-CimMethod -MethodName Terminate" >nul
echo Lab Monitor stopped.
