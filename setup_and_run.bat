@echo off
cd /d %~dp0
setlocal

echo ===================================================
echo      Chatterbox Select-and-Read Bot Setup
echo ===================================================

REM Check for Python 3.10
py -3.10 --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python 3.10 not found. Installing via winget...
    winget install -e --id Python.Python.3.10 --accept-package-agreements --accept-source-agreements
    if %errorlevel% neq 0 (
        echo Failed to install Python 3.10.
        echo Please install it manually from python.org.
        pause
        exit /b 1
    )
    echo Python 3.10 installed!
)

REM Create venv if needed
if not exist "venv_py310" (
    echo Creating virtual environment...
    py -3.10 -m venv venv_py310
)

echo Installing dependencies...
venv_py310\Scripts\python -m pip install --upgrade pip
venv_py310\Scripts\pip install numpy torch torchaudio numba
venv_py310\Scripts\pip install -e .
venv_py310\Scripts\pip install keyboard pyperclip

echo.
echo ===================================================
echo Setup Complete! Starting Reader Bot...
echo Select text and press Ctrl+Alt+R to read.
echo ===================================================
echo.

venv_py310\Scripts\python reader_bot.py
pause
