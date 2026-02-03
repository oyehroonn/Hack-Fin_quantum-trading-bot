#!/bin/bash
# Fix pytz installation in API venv

cd "$(dirname "$0")"

echo "Checking API virtual environment..."

if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

echo "Activating virtual environment..."
source venv/bin/activate

echo "Installing/updating pytz..."
pip install --upgrade pytz

echo "Verifying pytz installation..."
python3 -c "import pytz; print(f'✓ pytz {pytz.__version__} installed successfully')"

echo ""
echo "✓ pytz is now installed. Please restart your backend server:"
echo "  cd api && source venv/bin/activate && uvicorn main:app --reload --port 8000"
