"""Ensemble blender: weighted combination of multiple model predictions.

Supports fixed weights, performance-based weights, and learned stacking.
"""

import numpy as np
from typing import Any, Optional

from loguru import logger

from models.base import TradingModel


class EnsembleBlender(TradingModel):
    """Weighted ensemble of multiple TradingModels.

    Blending methods:
      - 'equal': equal weight to all models
      - 'performance': weight by recent Sharpe/accuracy
      - 'stacking': train a meta-model (logistic regression) on model outputs
    """

    model_type = "ensemble_blender"

    def __init__(
        self,
        models: list[TradingModel],
        model_id: str = "ensemble_blender",
        version: str = "1.0.0",
        method: str = "equal",
        weights: Optional[list[float]] = None,
        **params: Any,
    ) -> None:
        super().__init__(model_id, version, **params)
        self.models = models
        self.method = method
        self._weights = weights
        self._meta_model = None

        if weights is not None:
            assert len(weights) == len(models), "Weights must match number of models"

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "EnsembleBlender":
        """Train all base models and optionally learn stacking weights.

        Args:
            X: Training features
            y: Training labels
            feature_names: Feature names
            X_val: Validation features (used for stacking)
            y_val: Validation labels (used for stacking)
        """
        # Train base models that aren't already fitted
        for i, model in enumerate(self.models):
            if not model.is_fitted:
                logger.info(f"Training base model {i}: {model.model_id}")
                if hasattr(model, 'fit'):
                    model.fit(X, y, feature_names, X_val=X_val, y_val=y_val)

        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]

        # Learn weights if stacking
        if self.method == "stacking" and X_val is not None and y_val is not None:
            self._fit_stacking(X_val, y_val)
        elif self.method == "performance" and X_val is not None and y_val is not None:
            self._fit_performance_weights(X_val, y_val)
        elif self._weights is None:
            # Equal weights
            n = len(self.models)
            self._weights = [1.0 / n] * n

        self.is_fitted = True
        logger.info(
            f"Ensemble blender fitted: {len(self.models)} models, "
            f"method={self.method}, weights={[f'{w:.3f}' for w in (self._weights or [])]}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Blend model predictions into a single probability."""
        if not self.is_fitted:
            raise RuntimeError("Ensemble not fitted")

        # Collect predictions from all models
        all_probas = np.column_stack([
            model.predict_proba(X) for model in self.models
        ])

        if self.method == "stacking" and self._meta_model is not None:
            return self._predict_stacking(all_probas)

        # Weighted average
        weights = np.array(self._weights or [1.0 / len(self.models)] * len(self.models))
        blended = all_probas @ weights

        return np.clip(blended, 0.01, 0.99)

    def _fit_stacking(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Learn stacking weights via logistic regression on model outputs."""
        all_probas = np.column_stack([
            model.predict_proba(X_val) for model in self.models
        ])

        try:
            from sklearn.linear_model import LogisticRegression
            meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
            meta.fit(all_probas, y_val)
            self._meta_model = meta

            # Extract effective weights from coefficients
            raw_weights = np.abs(meta.coef_[0])
            self._weights = (raw_weights / raw_weights.sum()).tolist()

        except ImportError:
            logger.warning("sklearn not available for stacking, using equal weights")
            self._weights = [1.0 / len(self.models)] * len(self.models)

    def _predict_stacking(self, all_probas: np.ndarray) -> np.ndarray:
        """Predict using the trained stacking meta-model."""
        return np.clip(self._meta_model.predict_proba(all_probas)[:, 1], 0.01, 0.99)

    def _fit_performance_weights(self, X_val: np.ndarray, y_val: np.ndarray) -> None:
        """Weight models by their accuracy on validation data."""
        accuracies = []
        for model in self.models:
            probas = model.predict_proba(X_val)
            preds = (probas > 0.5).astype(int)
            acc = float(np.mean(preds == y_val))
            # Penalize models worse than random
            accuracies.append(max(acc - 0.5, 0.01))

        total = sum(accuracies)
        self._weights = [a / total for a in accuracies]

    def get_model_weights(self) -> dict[str, float]:
        """Get the weight assigned to each model."""
        return {
            model.model_id: w
            for model, w in zip(self.models, self._weights or [])
        }
