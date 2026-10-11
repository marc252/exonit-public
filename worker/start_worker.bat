@echo off
title ExoNit Worker
call ..\venv\Scripts\activate
python worker.py --config config.json
pause
