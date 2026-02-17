#!/usr/bin/env python3
"""
Launcher for the backend. Use this instead of python main.py.
Uses the project venv so dependencies (loguru, etc.) are found.

  python3 api/run.py
  # or from api/:  python3 run.py
"""
import subprocess
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
venv_python = project_root / "api" / "venv" / "bin" / "python"

if not venv_python.exists():
    print("ERROR: Virtual environment not found. Run first:")
    print("  python3 -m venv api/venv")
    print("  api/venv/bin/pip install -r api/requirements.txt")
    print("  api/venv/bin/pip install -e .")
    print("\nOr from project root:  ./start_backend.sh")
    sys.exit(1)

result = subprocess.run(
    [
        str(venv_python),
        "-m",
        "uvicorn",
        "api.main:app",
        "--reload",
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
    ],
    cwd=str(project_root),
)
sys.exit(result.returncode)
