"""Experiment tracking for research runs.

Stores experiment configs, results, and metadata in JSON files
for reproducibility and comparison.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from core.types import ExperimentResult, ModelMetrics


class ExperimentTracker:
    """Tracks research experiments: training runs, evaluations, comparisons."""

    def __init__(self, base_dir: str = "experiments") -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._index_file = self.base_dir / "index.json"
        self._index = self._load_index()

    def start_experiment(
        self,
        name: str,
        config: dict[str, Any],
        tags: Optional[list[str]] = None,
    ) -> str:
        """Start a new experiment and return its ID.

        Args:
            name: Human-readable experiment name
            config: Full configuration dict (model params, features, data, etc.)
            tags: Optional tags for filtering

        Returns:
            Experiment ID
        """
        exp_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"

        entry = {
            "experiment_id": exp_id,
            "name": name,
            "config": config,
            "tags": tags or [],
            "started_at": datetime.now().isoformat(),
            "status": "running",
            "metrics": {},
        }

        # Save experiment
        exp_dir = self.base_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)
        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(entry, f, indent=2, default=str)

        # Update index
        self._index[exp_id] = {
            "name": name,
            "started_at": entry["started_at"],
            "status": "running",
            "tags": tags or [],
        }
        self._save_index()

        logger.info(f"Started experiment: {exp_id}")
        return exp_id

    def log_metrics(
        self,
        experiment_id: str,
        metrics: ModelMetrics,
        phase: str = "test",
    ) -> None:
        """Log metrics for a phase of the experiment.

        Args:
            experiment_id: Experiment ID
            metrics: ModelMetrics to log
            phase: Phase name ('train', 'val', 'test', 'walk_forward')
        """
        exp_dir = self.base_dir / experiment_id
        if not exp_dir.exists():
            logger.warning(f"Experiment {experiment_id} not found")
            return

        # Load experiment
        with open(exp_dir / "experiment.json") as f:
            entry = json.load(f)

        # Add metrics
        entry["metrics"][phase] = {
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "total_return": metrics.total_return,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "num_trades": metrics.num_trades,
            "stability": metrics.stability,
            "turnover": metrics.turnover,
            "calmar": metrics.calmar,
            **metrics.extra,
        }

        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(entry, f, indent=2, default=str)

    def complete_experiment(
        self,
        experiment_id: str,
        status: str = "completed",
        notes: str = "",
    ) -> None:
        """Mark an experiment as complete."""
        exp_dir = self.base_dir / experiment_id
        if not exp_dir.exists():
            return

        with open(exp_dir / "experiment.json") as f:
            entry = json.load(f)

        entry["status"] = status
        entry["completed_at"] = datetime.now().isoformat()
        entry["notes"] = notes

        with open(exp_dir / "experiment.json", "w") as f:
            json.dump(entry, f, indent=2, default=str)

        if experiment_id in self._index:
            self._index[experiment_id]["status"] = status
            self._save_index()

        logger.info(f"Experiment {experiment_id} → {status}")

    def get_experiment(self, experiment_id: str) -> Optional[dict[str, Any]]:
        """Load a single experiment."""
        exp_file = self.base_dir / experiment_id / "experiment.json"
        if not exp_file.exists():
            return None
        with open(exp_file) as f:
            return json.load(f)

    def list_experiments(
        self,
        tags: Optional[list[str]] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """List experiments with optional filtering."""
        results = []
        for exp_id, info in sorted(self._index.items(), key=lambda x: x[1].get("started_at", ""), reverse=True):
            if status and info.get("status") != status:
                continue
            if tags and not any(t in info.get("tags", []) for t in tags):
                continue
            results.append({"experiment_id": exp_id, **info})
            if len(results) >= limit:
                break
        return results

    def compare_experiments(
        self,
        experiment_ids: list[str],
        metric: str = "sharpe",
        phase: str = "test",
    ) -> list[dict[str, Any]]:
        """Compare experiments side by side on a metric."""
        comparisons = []
        for exp_id in experiment_ids:
            exp = self.get_experiment(exp_id)
            if exp is None:
                continue
            metrics = exp.get("metrics", {}).get(phase, {})
            comparisons.append({
                "experiment_id": exp_id,
                "name": exp.get("name", ""),
                metric: metrics.get(metric, None),
                "all_metrics": metrics,
            })

        comparisons.sort(key=lambda x: x.get(metric) or 0, reverse=True)
        return comparisons

    def _load_index(self) -> dict[str, Any]:
        if self._index_file.exists():
            with open(self._index_file) as f:
                return json.load(f)
        return {}

    def _save_index(self) -> None:
        with open(self._index_file, "w") as f:
            json.dump(self._index, f, indent=2, default=str)
