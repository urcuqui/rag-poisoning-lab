#!/bin/bash

LAB_DIR="/opt/rag-poisoning-lab"
VENV_DIR="$LAB_DIR/venv"
APP="app.py"

echo "[*] Starting RAG Poisoning Lab..."

# 1.Verify the lab directory exists
if [ ! -d "$LAB_DIR" ]; then
  echo "[!] Lab directory not found: $LAB_DIR"
  exit 1
fi

cd "$LAB_DIR" || exit 1

# 2. Activate virtual environment
if [ -d "$VENV_DIR" ]; then
  echo "[*] Activating virtual environment..."
  source "$VENV_DIR/bin/activate"
else
  echo "[!] Virtual environment not found."
  echo "[!] Expected at: $VENV_DIR"
  exit 1
fi

# 3. Execute the application
if [ -f "$APP" ]; then
  echo "[*] Running application..."
  python "$APP"
else
  echo "[!] Application file not found: $APP"
  exit 1
fi
