#!/bin/bash

# Restart Backend Server Script
# This kills any existing server and starts a fresh one

cd "$(dirname "$0")"

echo "🔄 Restarting Backend Server..."
echo ""

# Kill any existing uvicorn process on port 8000
echo "🛑 Stopping existing server..."
lsof -ti:8000 | xargs kill -9 2>/dev/null && echo "   Server stopped" || echo "   No server running"

# Clear Python cache
echo "🧹 Clearing Python cache..."
find . -type d -name __pycache__ -exec rm -r {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null
echo "   Cache cleared"

# Wait a moment
sleep 1

# Check if venv exists
if [ ! -d "api/venv" ]; then
    echo "❌ Virtual environment not found!"
    echo "   Please run: cd api && ./setup_backend.sh"
    exit 1
fi

# Activate venv and start server
echo "🚀 Starting backend server..."
source api/venv/bin/activate

# Make sure main project is installed
pip install -e . > /dev/null 2>&1

# Start server from project root
echo "   Server starting on http://localhost:8000"
echo "   Press Ctrl+C to stop"
echo ""

uvicorn api.main:app --reload --port 8000
