#!/bin/bash

# Start Backend Server Script
# This script sets up and starts the FastAPI backend

cd "$(dirname "$0")"

echo "🚀 Starting Backend Server..."
echo ""

# Check if venv exists, if not create it
if [ ! -d "api/venv" ]; then
    echo "📦 Virtual environment not found. Creating it..."
    cd api
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
    cd ..
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate venv and install main project
echo "📦 Installing main project..."
cd api
source venv/bin/activate
cd ..

# Install main project in editable mode
pip install -e .

echo ""
echo "✅ Dependencies installed"
echo ""
echo "🌐 Starting backend server on http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

# Start the server (from project root, so we can use api.main:app)
source api/venv/bin/activate
uvicorn api.main:app --reload --port 8000
