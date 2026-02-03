#!/bin/bash

# Fix API dependencies script
# This ensures all dependencies are installed in the API venv

cd "$(dirname "$0")/.."

echo "🔧 Fixing API dependencies..."
echo ""

# Check if venv exists
if [ ! -d "api/venv" ]; then
    echo "📦 Creating virtual environment..."
    cd api
    python3 -m venv venv
    cd ..
fi

# Activate venv
echo "📦 Installing dependencies..."
source api/venv/bin/activate

# Upgrade pip
pip install --upgrade pip > /dev/null 2>&1

# Install requirements
pip install -r api/requirements.txt

# Install main project in editable mode (CRITICAL - this ensures all modules are available)
echo "📦 Installing main project in editable mode..."
pip install -e .

echo ""
echo "✅ Dependencies installed!"
echo ""
echo "To start the backend:"
echo "  cd /Users/oyehroonn/Downloads/Hack-Fin_quantum-trading-bot"
echo "  source api/venv/bin/activate"
echo "  uvicorn api.main:app --reload --port 8000"
