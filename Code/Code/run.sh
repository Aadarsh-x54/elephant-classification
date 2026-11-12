#!/usr/bin/env bash
# Quickstart script for Linux/macOS
set -e

echo "=== Elephant Classifier Setup ==="
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "Running tests..."
pytest tests/test_project.py -v

echo "Starting Streamlit web app..."
python -m streamlit run app.py

