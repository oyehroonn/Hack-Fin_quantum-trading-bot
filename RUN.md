# How to Run the Quantum Trading Bot

## Backend (API Server)

**Option 1 — Recommended (uses venv automatically):**
```bash
# From project root
python3 api/run.py
```

**Option 2 — Shell script:**
```bash
# From project root
./start_backend.sh
```

**Option 3 — Manual:**
```bash
# From project root
source api/venv/bin/activate
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

> **Note:** Don't run `python3 main.py` from the `api/` folder — that uses system Python which may not have `loguru` and other deps. Use `api/run.py` or the venv.

**First-time setup:** If the venv doesn't exist:
```bash
python3 -m venv api/venv
api/venv/bin/pip install -r api/requirements.txt
api/venv/bin/pip install -e .
```

---

## Frontend

```bash
cd frontend
npm install
npm run dev
```

Open http://localhost:5173

---

## Quick Start

1. Start backend: `python3 api/run.py`
2. Start frontend: `cd frontend && npm run dev`
3. Open http://localhost:5173
4. Use **Trading Terminal** for live suggestions, BUY/SELL, and P&L
5. Use **Backtest** for historical strategy testing
