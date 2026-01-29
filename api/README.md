# Backend API Setup

## Quick Setup

### Option 1: Automated Setup (Recommended)

```bash
cd api
./setup_backend.sh
```

### Option 2: Manual Setup

```bash
# Navigate to api directory
cd api

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install API dependencies
pip install -r requirements.txt

# Install main project in editable mode (so API can import backtest modules)
cd ..
pip install -e .
cd api
```

## Running the Server

After setup, start the server:

```bash
# Make sure virtual environment is activated
source venv/bin/activate  # Windows: venv\Scripts\activate

# Start the server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

## Troubleshooting

### "Command not found" errors

- Make sure you're using `python3` instead of `python`
- On some systems, you may need to use `python3 -m venv` instead of `python -m venv`
- Make sure Python 3.11+ is installed: `python3 --version`

### Import errors when running uvicorn

- Make sure you installed the main project: `cd .. && pip install -e . && cd api`
- The API needs access to the `backtest` module from the main project

### Port already in use

- Change the port: `uvicorn main:app --reload --port 8001`
- Or kill the process using port 8000
