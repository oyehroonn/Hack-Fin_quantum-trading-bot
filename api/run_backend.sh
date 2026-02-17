#!/bin/bash
# Run this from api/ OR project root. Uses venv and starts the backend.

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "🚀 Starting Quantum Trading Bot Backend..."
echo ""

if [ ! -d "api/venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv api/venv
    echo "📦 Installing dependencies..."
    source api/venv/bin/activate
    pip install --upgrade pip
    pip install -r api/requirements.txt
    pip install -e .
else
    source api/venv/bin/activate
fi

# Ensure deps are installed
pip install -q -r api/requirements.txt
pip install -q -e . 2>/dev/null || true

echo "✅ Backend ready. Starting on http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"
echo "   Press Ctrl+C to stop"
echo ""

exec uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
