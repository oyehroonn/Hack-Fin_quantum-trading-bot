"""XGBoost-based directional classifier for trading.

Predicts P(price_up | features) using gradient boosted trees.
Supports class weighting, early stopping, and feature importance.
"""

import numpy as np
from typing import Any, Optional

from loguru import logger

from models.base import TradingModel


class XGBDirectionClassifier(TradingModel):
    """XGBoost classifier for predicting price direction."""

    model_type = "ml_xgb_classifier"

    def __init__(
        self,
        model_id: str = "xgb_direction",
        version: str = "1.0.0",
        n_estimators: int = 200,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 5,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        scale_pos_weight: float = 1.0,
        early_stopping_rounds: int = 20,
        random_state: int = 42,
        **params: Any,
    ) -> None:
        super().__init__(model_id, version, **params)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.scale_pos_weight = scale_pos_weight
        self.early_stopping_rounds = early_stopping_rounds
        self.random_state = random_state
        self._model = None
        self._feature_importance: Optional[np.ndarray] = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "XGBDirectionClassifier":
        """Train the XGBoost classifier.

        Args:
            X: Training features (n_samples, n_features)
            y: Binary labels (0 = down, 1 = up)
            feature_names: Feature column names
            X_val: Validation features for early stopping
            y_val: Validation labels
        """
        try:
            from xgboost import XGBClassifier
        except ImportError:
            # Fallback to sklearn gradient boosting
            from sklearn.ensemble import GradientBoostingClassifier
            logger.warning("xgboost not installed, falling back to sklearn GradientBoosting")
            self._model = GradientBoostingClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            self._model.fit(X, y)
            self.is_fitted = True
            self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
            self._feature_importance = self._model.feature_importances_
            logger.info(f"Trained GBM fallback on {X.shape[0]} samples, {X.shape[1]} features")
            return self

        self._model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_weight=self.min_child_weight,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            scale_pos_weight=self.scale_pos_weight,
            random_state=self.random_state,
            eval_metric="logloss",
            use_label_encoder=False,
            verbosity=0,
        )

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            fit_params["eval_set"] = [(X_val, y_val)]
            fit_params["verbose"] = False

        # Handle NaN/inf in features
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        if X_val is not None:
            X_val_clean = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            fit_params["eval_set"] = [(X_val_clean, y_val)]

        self._model.fit(X_clean, y, **fit_params)
        self.is_fitted = True
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._feature_importance = self._model.feature_importances_

        logger.info(
            f"Trained XGB on {X.shape[0]} samples, {X.shape[1]} features, "
            f"best iteration: {self._model.best_iteration if hasattr(self._model, 'best_iteration') else 'N/A'}"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict P(up) for each sample."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        probas = self._model.predict_proba(X_clean)

        # Return P(positive class)
        if probas.ndim == 2:
            return probas[:, 1]
        return probas

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        if self._feature_importance is None:
            return {}
        return {
            name: float(imp)
            for name, imp in zip(self._feature_names, self._feature_importance)
        }

    def get_params(self) -> dict[str, Any]:
        return {
            **super().get_params(),
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
        }
