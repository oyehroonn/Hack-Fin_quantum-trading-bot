# Trading Bot UI Setup

This guide will help you set up and run the FastAPI + React UI for the trading bot.

## Prerequisites

- Python 3.11+
- Node.js 18+ and npm
- Poetry (for Python dependencies)

## Quick Start

### 1. Backend Setup

**Option A: Automated Setup (Recommended)**
```bash
cd api
./setup_backend.sh
```

**Option B: Manual Setup**
```bash
# Navigate to API directory
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

# Install main project in editable mode (IMPORTANT!)
cd ..
pip install -e .
cd api

# Run the API server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### 2. Frontend Setup

Open a new terminal:

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm run dev
```

The UI will be available at `http://localhost:5173`

## Usage

1. **Open the UI**: Navigate to `http://localhost:5173` in your browser

2. **Test with Synthetic Data**:
   - Check "Use Synthetic Data (for testing)"
   - Configure parameters (Initial Cash, Fast/Slow SMA periods)
   - Click "Run Backtest"

3. **Test with Real Data**:
   - Uncheck "Use Synthetic Data"
   - Upload a CSV or Parquet file with columns: `timestamp`, `symbol`, `open`, `high`, `low`, `close`, `volume`
   - Configure parameters
   - Click "Run Backtest"

## API Endpoints

- `GET /` - API health check
- `POST /api/backtest` - Run backtest with uploaded file
- `POST /api/backtest/synthetic` - Run backtest with synthetic data

## Example CSV Format

```csv
timestamp,symbol,open,high,low,close,volume
2024-01-01,AAPL,100.0,101.0,99.0,100.0,1000
2024-01-02,AAPL,100.0,102.0,99.5,101.0,1100
```

## Troubleshooting

### Backend Issues

- **Import errors**: Make sure you're in the project root when running the API, or adjust Python path
- **Port already in use**: Change the port in the uvicorn command: `--port 8001`

### Frontend Issues

- **CORS errors**: Make sure the backend is running and CORS is configured correctly
- **Proxy errors**: Check that `vite.config.js` has the correct backend URL

### Data Issues

- **File format**: Ensure CSV has required columns (timestamp, open, high, low, close, volume)
- **Date format**: Timestamps should be parseable by pandas (ISO format recommended)

## Development

### Backend Development

The FastAPI server supports hot-reload with `--reload` flag. Changes to `api/main.py` will automatically restart the server.

### Frontend Development

Vite supports hot module replacement (HMR). Changes to React components will update in the browser without full page reload.

## Production Build

### Frontend

```bash
cd frontend
npm run build
```

The built files will be in `frontend/dist/`. You can serve these with any static file server.

### Backend

For production, use a production ASGI server like Gunicorn with Uvicorn workers:

```bash
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker
```
