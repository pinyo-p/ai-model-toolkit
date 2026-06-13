#!/bin/bash
set -e

echo "=== AI Toolkit Updater ==="

# Pull latest
echo "Pulling latest changes..."
git pull origin main

# Activate venv if exists
if [ -d "venv" ]; then
    source venv/bin/activate
    echo "Installing any new dependencies..."
    pip install -r requirements.txt -q
else
    echo "venv not found. Run install.bash first."
    exit 1
fi

echo "=== Done ==="
echo "Run: source venv/bin/activate && python main.py"
