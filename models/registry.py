"""Model registry: save/load/promote models with champion/challenger tracking.

Storage layout:
    models_registry/
    ├── {model_id}/
    │   ├── {version}/
    │   │   ├── artifact.joblib
    │   │   ├── metadata.json
    │   │   └── metrics.json
    │   └── latest -> {version}/   (symlink concept via metadata)
    └── champions.json              (asset_class/timeframe → model_id/version)
"""

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
from loguru import logger

from core.interfaces import ModelRegistry as ModelRegistryInterface
from core.types import ModelMetrics


class FileModelRegistry(ModelRegistryInterface):
    """File-based model registry with champion/challenger management."""

    def __init__(self, base_path: str = "models_registry") -> None:
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self._champions_file = self.base_path / "champions.json"
        self._champions: dict[str, dict[str, str]] = self._load_champions()

    # ── Persistence ──

    def save_model(
        self,
        model_id: str,
        version: str,
        artifact: Any,
        metadata: dict[str, Any],
    ) -> None:
        """Save a model artifact + metadata to disk."""
        model_dir = self.base_path / model_id / version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save artifact
        artifact_path = model_dir / "artifact.joblib"
        joblib.dump(artifact, artifact_path, compress=3)

        # Save metadata
        meta = {
            "model_id": model_id,
            "version": version,
            "saved_at": datetime.now().isoformat(),
            **metadata,
        }
        meta_path = model_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2, default=str)

        # Update latest pointer
        latest_file = self.base_path / model_id / "latest.json"
        with open(latest_file, "w") as f:
            json.dump({"version": version}, f)

        logger.info(f"Saved model {model_id} v{version} to {model_dir}")

    def load_model(self, model_id: str, version: Optional[str] = None) -> Any:
        """Load a model artifact. Uses latest version if not specified."""
        if version is None:
            version = self._get_latest_version(model_id)
            if version is None:
                raise FileNotFoundError(f"No versions found for model {model_id}")

        artifact_path = self.base_path / model_id / version / "artifact.joblib"
        if not artifact_path.exists():
            raise FileNotFoundError(f"Model artifact not found: {artifact_path}")

        return joblib.load(artifact_path)

    def load_metadata(self, model_id: str, version: Optional[str] = None) -> dict[str, Any]:
        """Load model metadata."""
        if version is None:
            version = self._get_latest_version(model_id)
            if version is None:
                raise FileNotFoundError(f"No versions found for model {model_id}")

        meta_path = self.base_path / model_id / version / "metadata.json"
        if not meta_path.exists():
            return {}
        with open(meta_path) as f:
            return json.load(f)

    def list_models(self, asset_class: Optional[str] = None) -> list[dict[str, Any]]:
        """List all registered models with their latest metadata."""
        models = []
        for model_dir in sorted(self.base_path.iterdir()):
            if not model_dir.is_dir() or model_dir.name.startswith("."):
                continue
            if model_dir.name == "champions.json":
                continue

            latest_version = self._get_latest_version(model_dir.name)
            if latest_version is None:
                continue

            meta = self.load_metadata(model_dir.name, latest_version)
            if asset_class and meta.get("asset_class") != asset_class:
                continue

            meta["latest_version"] = latest_version
            meta["versions"] = [
                v.name for v in model_dir.iterdir()
                if v.is_dir() and not v.name.startswith(".")
            ]
            models.append(meta)

        return models

    def list_versions(self, model_id: str) -> list[str]:
        """List all versions for a model."""
        model_dir = self.base_path / model_id
        if not model_dir.exists():
            return []
        return sorted([
            v.name for v in model_dir.iterdir()
            if v.is_dir() and not v.name.startswith(".")
        ])

    def delete_model(self, model_id: str, version: Optional[str] = None) -> None:
        """Delete a model (all versions) or a specific version."""
        if version:
            target = self.base_path / model_id / version
        else:
            target = self.base_path / model_id

        if target.exists():
            shutil.rmtree(target)
            logger.info(f"Deleted {target}")

    # ── Champion / Challenger ──

    def get_champion(self, asset_class: str, timeframe: str) -> Optional[str]:
        """Get the champion model_id for a given asset_class/timeframe."""
        key = f"{asset_class}/{timeframe}"
        entry = self._champions.get(key)
        return entry["model_id"] if entry else None

    def get_champion_info(self, asset_class: str, timeframe: str) -> Optional[dict[str, str]]:
        """Get full champion info (model_id + version)."""
        key = f"{asset_class}/{timeframe}"
        return self._champions.get(key)

    def promote(self, model_id: str, version: str, asset_class: str, timeframe: str) -> None:
        """Promote a model version to champion for a given asset_class/timeframe."""
        key = f"{asset_class}/{timeframe}"
        self._champions[key] = {
            "model_id": model_id,
            "version": version,
            "promoted_at": datetime.now().isoformat(),
        }
        self._save_champions()
        logger.info(f"Promoted {model_id} v{version} as champion for {key}")

    # ── Metrics ──

    def record_metrics(self, metrics: ModelMetrics) -> None:
        """Record evaluation metrics for a model version."""
        model_dir = self.base_path / metrics.model_id / metrics.model_version
        model_dir.mkdir(parents=True, exist_ok=True)

        metrics_file = model_dir / f"metrics_{metrics.evaluation_type}.json"

        # Load existing or start fresh
        existing = []
        if metrics_file.exists():
            with open(metrics_file) as f:
                existing = json.load(f)

        entry = {
            "timestamp": metrics.timestamp.isoformat(),
            "evaluation_type": metrics.evaluation_type,
            "sharpe": metrics.sharpe,
            "sortino": metrics.sortino,
            "total_return": metrics.total_return,
            "max_drawdown": metrics.max_drawdown,
            "win_rate": metrics.win_rate,
            "profit_factor": metrics.profit_factor,
            "num_trades": metrics.num_trades,
            "avg_trade_pnl": metrics.avg_trade_pnl,
            "turnover": metrics.turnover,
            "calmar": metrics.calmar,
            "stability": metrics.stability,
            **metrics.extra,
        }
        existing.append(entry)

        with open(metrics_file, "w") as f:
            json.dump(existing, f, indent=2, default=str)

    def get_metrics(
        self,
        model_id: str,
        version: str,
        evaluation_type: str = "oos",
    ) -> list[dict[str, Any]]:
        """Get evaluation metrics for a model version."""
        metrics_file = self.base_path / model_id / version / f"metrics_{evaluation_type}.json"
        if not metrics_file.exists():
            return []
        with open(metrics_file) as f:
            return json.load(f)

    # ── Promotion Logic ──

    def evaluate_challenger(
        self,
        challenger_metrics: ModelMetrics,
        asset_class: str,
        timeframe: str,
        min_trades: int = 30,
        max_turnover: float = 6.0,
        require_lower_drawdown: bool = True,
    ) -> tuple[bool, str]:
        """Evaluate whether a challenger should replace the current champion.

        Returns:
            (should_promote, reason)
        """
        # Basic viability checks
        if challenger_metrics.num_trades < min_trades:
            return False, f"Too few trades ({challenger_metrics.num_trades} < {min_trades})"
        if challenger_metrics.turnover > max_turnover:
            return False, f"Turnover too high ({challenger_metrics.turnover:.2f} > {max_turnover})"

        # Get champion metrics
        champ_info = self.get_champion_info(asset_class, timeframe)
        if champ_info is None:
            return True, "No existing champion — auto-promote"

        champ_metrics_list = self.get_metrics(
            champ_info["model_id"],
            champ_info["version"],
            evaluation_type="oos",
        )
        if not champ_metrics_list:
            return True, "Champion has no recorded OOS metrics — promote challenger"

        champ = champ_metrics_list[-1]  # Latest metrics

        # Comparison
        reasons = []

        if challenger_metrics.sharpe <= champ.get("sharpe", 0):
            reasons.append(
                f"Sharpe not better ({challenger_metrics.sharpe:.3f} <= {champ.get('sharpe', 0):.3f})"
            )

        if require_lower_drawdown and challenger_metrics.max_drawdown > champ.get("max_drawdown", 1.0):
            reasons.append(
                f"Drawdown worse ({challenger_metrics.max_drawdown:.3f} > {champ.get('max_drawdown', 1.0):.3f})"
            )

        if reasons:
            return False, "; ".join(reasons)

        return True, (
            f"Challenger wins: Sharpe {challenger_metrics.sharpe:.3f} vs {champ.get('sharpe', 0):.3f}, "
            f"DD {challenger_metrics.max_drawdown:.3f} vs {champ.get('max_drawdown', 1.0):.3f}"
        )

    # ── Internal ──

    def _get_latest_version(self, model_id: str) -> Optional[str]:
        latest_file = self.base_path / model_id / "latest.json"
        if latest_file.exists():
            with open(latest_file) as f:
                data = json.load(f)
                return data.get("version")
        # Fallback: pick highest version dir
        model_dir = self.base_path / model_id
        if not model_dir.exists():
            return None
        versions = [
            v.name for v in model_dir.iterdir()
            if v.is_dir() and not v.name.startswith(".")
        ]
        return sorted(versions)[-1] if versions else None

    def _load_champions(self) -> dict[str, dict[str, str]]:
        if self._champions_file.exists():
            with open(self._champions_file) as f:
                return json.load(f)
        return {}

    def _save_champions(self) -> None:
        with open(self._champions_file, "w") as f:
            json.dump(self._champions, f, indent=2, default=str)
