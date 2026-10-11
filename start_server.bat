@echo off
title ExoNit Server
cd /d "%~dp0"
if exist venv\Scripts\activate.bat (
    call venv\Scripts\activate
)
python run_nightly.py serve
pause
