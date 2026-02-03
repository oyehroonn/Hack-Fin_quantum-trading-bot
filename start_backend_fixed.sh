#!/bin/bash
# Start backend with proper setup

cd "$(dirname "$0")"

echo "🚀 Starting Trading Bot Backend..."
echo ""

# Check if venv exists
if [ ! -d "api/venv" ]; then
    echo "Creating virtual environment..."
    cd api
    python3 -m venv venv
    cd ..
fi

# Activate venv
echo "Activating virtual environment..."
source api/venv/bin/activate

# Install/upgrade pytz and other dependencies
echo "Installing dependencies..."
pip install --upgrade pytz pandas numpy loguru fastapi uvicorn yfinance requests pydantic pydantic-settings pyyaml pyarrow scipy scikit-learn 2>&1 | grep -E "(Requirement|Successfully|already satisfied)" || true

# Install main project in editable mode
echo "Installing main project..."
pip install -e . 2>&1 | grep -E "(Requirement|Successfully|already satisfied)" || true

# Verify pytz
echo ""
echo "Verifying pytz installation..."
python3 -c "import pytz; print(f'✓ pytz {pytz.__version__} installed')" || {
    echo "✗ pytz installation failed!"
    exit 1
}

echo ""
echo "✓ All dependencies installed"
echo ""
echo "Starting backend server on http://localhost:8000"
echo "Press Ctrl+C to stop"
echo ""

# Start uvicorn
cd "$(dirname "$0")"
uvicorn api.main:app --reload --port 8000
