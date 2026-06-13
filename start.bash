#!/bin/bash
set -e

echo "=== AI Toolkit Starter ==="

if [ ! -d "venv" ]; then
    echo "venv not found. Run install.bash first."
    exit 1
fi

source venv/bin/activate
echo "Starting server on http://localhost:7800"
python main.py
