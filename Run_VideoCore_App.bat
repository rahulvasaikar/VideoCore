@echo off
title VideoCore Studio
echo ========================================================
echo               Starting VideoCore Studio...
echo ========================================================
echo.

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python virtual environment not found in .venv!
    echo Please ensure the virtual environment is created and configured.
    pause
    exit /b 1
)

echo Checking and freeing port 8501...
powershell -NoProfile -Command "Get-NetTCPConnection -LocalPort 8501 -ErrorAction SilentlyContinue | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue }" >nul 2>&1
for /f "tokens=5" %%a in ('netstat -aon ^| findstr /r /c:":8501 *LISTENING"') do (
    taskkill /F /PID %%a >nul 2>&1
)

echo Activating virtual environment and launching Web App...
echo App will open in your default browser at http://localhost:8501
echo.

.\.venv\Scripts\python.exe -m videocore ui

if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] VideoCore Studio encountered an error.
    pause
)
