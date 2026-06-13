@echo off
echo === AI Toolkit Starter ===

if not exist "venv" (
    echo venv not found. Run install.bat first.
    exit /b 1
)

call venv\Scripts\activate.bat
echo Starting server on http://localhost:7800
python main.py
