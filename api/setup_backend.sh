#!/bin/bash

# Setup script for backend API
# This installs all dependencies needed to run the FastAPI server

echo "🔧 Setting up backend API..."
echo ""

# Check if we're in the right directory
if [ ! -f "requirements.txt" ]; then
    echo "❌ Error: requirements.txt not found. Make sure you're in the api/ directory"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📥 Installing dependencies (this may take a few minutes)..."
pip install -r requirements.txt

# Install the main project in editable mode (so API can import backtest modules)
echo "📦 Installing main project in editable mode..."
cd ..
pip install -e .
cd api

echo ""
echo "✅ Backend setup complete!"
echo ""
echo "To start the server, run:"
echo "  source venv/bin/activate"
echo "  uvicorn main:app --reload --port 8000"
echo ""
