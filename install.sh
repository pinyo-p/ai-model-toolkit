#!/bin/bash
set -e

echo "=== AI Toolkit Installer ==="

# Check python3
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.10+ first."
    exit 1
fi

PYTHON=python3
echo "Using: $($PYTHON --version)"

# Create venv
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    $PYTHON -m venv venv
else
    echo "venv already exists, skipping."
fi

# Activate
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip -q

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=== Done ==="
echo "To run:"
echo "  source venv/bin/activate"
echo "  python main.py"
echo "Then open http://localhost:7800"
