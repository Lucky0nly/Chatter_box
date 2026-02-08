@echo off
cd /d %~dp0
echo Starting Chatterbox Reader Bot (Manual)...
venv_py310\Scripts\python reader_bot.py
pause
