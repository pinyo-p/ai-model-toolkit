@echo off
echo === AI Toolkit Updater ===

echo Pulling latest changes...
git pull origin main

if not exist "venv" (
    echo venv not found. Run install.bat first.
    exit /b 1
)

call venv\Scripts\activate.bat
echo Installing any new dependencies...
pip install -r requirements.txt -q

echo === Done ===
echo Run: venv\Scripts\activate ^&^& python main.py
