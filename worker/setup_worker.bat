@echo off
title ExoNit Worker Setup
echo ============================================================
echo  ExoNit Worker Installer
echo ============================================================
echo.
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Installing...
    winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
    echo Please restart this script after installation.
    pause
    exit /b
)
echo [OK] Python found
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate
echo Installing dependencies...
pip install -q numpy requests onnxruntime lightkurve
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [OK] NVIDIA GPU detected
    echo Installing CUDA BLS acceleration...
    pip install -q cupy-cuda12x
    set GPU_ENABLED=true
) else (
    echo [INFO] No GPU - CPU mode
    set GPU_ENABLED=false
)
echo.
set /p SERVER_URL="Server URL (e.g. http://YOUR_SERVER:8400): "
set /p TOKEN="Worker token (default: exonit-default-token): "
if "%TOKEN%"=="" set TOKEN=exonit-default-token
set /p WORKER_NAME="Worker name: "
if "%WORKER_NAME%"=="" set WORKER_NAME=worker-1
echo {"server_url":"%SERVER_URL%","token":"%TOKEN%","worker_name":"%WORKER_NAME%","cpu_threads":4,"gpu_enabled":%GPU_ENABLED%,"poll_interval_seconds":5,"heartbeat_interval_seconds":30,"fits_cache_dir":"fits_cache","gui_port":8401}> config.json
echo.
echo [OK] Setup complete! Run start_worker.bat to start.
echo.
echo Optional config flag (worker\config.json):
echo   ffi_fallback_on_load_fail: false  ^(set to true to recover load_failed TICs^)
pause
