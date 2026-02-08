@echo off
setlocal
cd /d "D:\chatterbox-master"
echo Starting Chatterbox Reader Bot (Manual)...
call venv_py310\Scripts\activate.bat
python reader_bot.py
endlocal
