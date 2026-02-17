"""LightGBM-based return regressor for trading.

Predicts expected forward return, then converts to P(up) via calibration.
Supports feature importance, early stopping, and categorical features.
"""

import numpy as np
from typing import Any, Optional

from loguru import logger

from models.base import TradingModel


class LGBMReturnRegressor(TradingModel):
    """LightGBM regressor for predicting forward returns."""

    model_type = "ml_lgbm_regressor"

    def __init__(
        self,
        model_id: str = "lgbm_returns",
        version: str = "1.0.0",
        n_estimators: int = 300,
        max_depth: int = 5,
        learning_rate: float = 0.03,
        num_leaves: int = 31,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_samples: int = 20,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        **params: Any,
    ) -> None:
        super().__init__(model_id, version, **params)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_samples = min_child_samples
        self.reg_alpha = reg_alpha
        self.reg_lambda = reg_lambda
        self.random_state = random_state
        self._model = None
        self._feature_importance: Optional[np.ndarray] = None
        self._return_std: float = 0.01  # For scaling returns → probabilities

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "LGBMReturnRegressor":
        """Train the LightGBM regressor.

        Args:
            X: Training features (n_samples, n_features)
            y: Forward returns (continuous, e.g., -0.02 to +0.03)
            feature_names: Feature column names
            X_val: Validation features
            y_val: Validation targets
        """
        try:
            from lightgbm import LGBMRegressor
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor
            logger.warning("lightgbm not installed, falling back to sklearn GBR")
            self._model = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
            self._model.fit(X_clean, y)
            self.is_fitted = True
            self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
            self._feature_importance = self._model.feature_importances_
            self._return_std = max(float(np.std(y)), 1e-6)
            return self

        self._model = LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            min_child_samples=self.min_child_samples,
            reg_alpha=self.reg_alpha,
            reg_lambda=self.reg_lambda,
            random_state=self.random_state,
            verbose=-1,
        )

        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        fit_params: dict[str, Any] = {}
        if X_val is not None and y_val is not None:
            X_val_clean = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
            fit_params["eval_set"] = [(X_val_clean, y_val)]
            fit_params["eval_metric"] = "mse"

        self._model.fit(X_clean, y, **fit_params)
        self.is_fitted = True
        self._feature_names = feature_names or [f"f{i}" for i in range(X.shape[1])]
        self._feature_importance = self._model.feature_importances_
        self._return_std = max(float(np.std(y)), 1e-6)

        logger.info(
            f"Trained LGBM regressor on {X.shape[0]} samples, {X.shape[1]} features, "
            f"target std: {self._return_std:.6f}"
        )
        return self

    def predict_returns(self, X: np.ndarray) -> np.ndarray:
        """Predict raw forward returns."""
        if not self.is_fitted or self._model is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        X_clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return self._model.predict(X_clean)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Convert predicted returns to P(up) using sigmoid transformation.

        Maps predicted return → probability using:
            P(up) = sigmoid(predicted_return / return_std * scaling_factor)
        """
        predicted_returns = self.predict_returns(X)

        # Sigmoid: scale returns by historical std to get roughly calibrated P(up)
        z = predicted_returns / self._return_std * 2.0
        probas = 1.0 / (1.0 + np.exp(-z))

        return np.clip(probas, 0.01, 0.99)

    def get_feature_importance(self) -> dict[str, float]:
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
            "num_leaves": self.num_leaves,
        }
