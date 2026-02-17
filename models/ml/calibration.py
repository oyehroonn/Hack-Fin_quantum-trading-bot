"""Probability calibration for trading models.

Raw model outputs often produce poorly calibrated probabilities.
This module wraps any TradingModel with isotonic or Platt calibration
so that P(up) = 0.6 actually means 60% of the time the price goes up.
"""

import numpy as np
from typing import Any, Optional

from loguru import logger

from models.base import TradingModel


class CalibratedModel(TradingModel):
    """Wraps a TradingModel with probability calibration.

    Uses either:
      - 'isotonic': non-parametric, more flexible (needs more data)
      - 'sigmoid': Platt scaling, parametric (works with less data)
    """

    model_type = "ml_calibrated"

    def __init__(
        self,
        base_model: TradingModel,
        method: str = "isotonic",
        model_id: Optional[str] = None,
        version: str = "1.0.0",
        **params: Any,
    ) -> None:
        mid = model_id or f"{base_model.model_id}_calibrated"
        super().__init__(mid, version, **params)
        self.base_model = base_model
        self.method = method
        self._calibrator = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: Optional[list[str]] = None,
        X_cal: Optional[np.ndarray] = None,
        y_cal: Optional[np.ndarray] = None,
    ) -> "CalibratedModel":
        """Two-stage fit: train base model, then calibrate.

        Args:
            X: Training features
            y: Training labels
            feature_names: Feature names
            X_cal: Calibration set features (if None, uses X)
            y_cal: Calibration set labels (if None, uses y)
        """
        # Stage 1: fit base model
        if not self.base_model.is_fitted:
            self.base_model.fit(X, y, feature_names)

        self._feature_names = self.base_model.feature_names

        # Stage 2: calibrate on held-out data
        cal_X = X_cal if X_cal is not None else X
        cal_y = y_cal if y_cal is not None else y

        raw_probas = self.base_model.predict_proba(cal_X)

        if self.method == "isotonic":
            self._calibrator = self._fit_isotonic(raw_probas, cal_y)
        elif self.method == "sigmoid":
            self._calibrator = self._fit_sigmoid(raw_probas, cal_y)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")

        self.is_fitted = True

        # Measure calibration improvement
        cal_probas = self._apply_calibration(raw_probas)
        raw_brier = float(np.mean((raw_probas - cal_y) ** 2))
        cal_brier = float(np.mean((cal_probas - cal_y) ** 2))

        logger.info(
            f"Calibrated {self.base_model.model_id}: "
            f"Brier score {raw_brier:.4f} → {cal_brier:.4f} "
            f"(method={self.method}, n_cal={len(cal_y)})"
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get calibrated probabilities."""
        raw = self.base_model.predict_proba(X)
        if self._calibrator is None:
            return raw
        return self._apply_calibration(raw)

    # ── Calibration implementations ──

    def _fit_isotonic(self, probas: np.ndarray, y: np.ndarray) -> Any:
        """Fit isotonic regression calibrator."""
        try:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
            ir.fit(probas, y)
            return {"type": "isotonic", "model": ir}
        except ImportError:
            logger.warning("sklearn not available for isotonic calibration, using binning fallback")
            return self._fit_binning(probas, y)

    def _fit_sigmoid(self, probas: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Fit Platt sigmoid calibrator (logistic regression on log-odds)."""
        # Platt scaling: fit a + b * log(p / (1-p)) to y
        eps = 1e-6
        log_odds = np.log(np.clip(probas, eps, 1 - eps) / np.clip(1 - probas, eps, 1 - eps))

        try:
            from sklearn.linear_model import LogisticRegression
            lr = LogisticRegression(C=1.0, solver="lbfgs")
            lr.fit(log_odds.reshape(-1, 1), y)
            return {"type": "sigmoid_sklearn", "model": lr}
        except ImportError:
            # Manual Platt scaling via least squares
            from numpy.linalg import lstsq
            A = np.column_stack([log_odds, np.ones_like(log_odds)])
            coeffs, _, _, _ = lstsq(A, y, rcond=None)
            return {"type": "sigmoid_manual", "a": float(coeffs[0]), "b": float(coeffs[1])}

    def _fit_binning(self, probas: np.ndarray, y: np.ndarray, n_bins: int = 10) -> dict[str, Any]:
        """Simple binning calibration as last-resort fallback."""
        bins = np.linspace(0, 1, n_bins + 1)
        bin_means = []
        bin_true_means = []

        for i in range(n_bins):
            mask = (probas >= bins[i]) & (probas < bins[i + 1])
            if mask.sum() > 0:
                bin_means.append(float(np.mean(probas[mask])))
                bin_true_means.append(float(np.mean(y[mask])))

        return {"type": "binning", "bin_means": bin_means, "bin_true_means": bin_true_means}

    def _apply_calibration(self, probas: np.ndarray) -> np.ndarray:
        """Apply fitted calibration to raw probabilities."""
        if self._calibrator is None:
            return probas

        cal_type = self._calibrator.get("type", "")

        if cal_type in ("isotonic", "sigmoid_sklearn"):
            model = self._calibrator["model"]
            if cal_type == "isotonic":
                return np.clip(model.predict(probas), 0.01, 0.99)
            else:
                eps = 1e-6
                log_odds = np.log(np.clip(probas, eps, 1 - eps) / np.clip(1 - probas, eps, 1 - eps))
                return np.clip(model.predict_proba(log_odds.reshape(-1, 1))[:, 1], 0.01, 0.99)

        elif cal_type == "sigmoid_manual":
            eps = 1e-6
            log_odds = np.log(np.clip(probas, eps, 1 - eps) / np.clip(1 - probas, eps, 1 - eps))
            z = self._calibrator["a"] * log_odds + self._calibrator["b"]
            return np.clip(1.0 / (1.0 + np.exp(-z)), 0.01, 0.99)

        elif cal_type == "binning":
            bm = np.array(self._calibrator["bin_means"])
            btm = np.array(self._calibrator["bin_true_means"])
            return np.clip(np.interp(probas, bm, btm), 0.01, 0.99)

        return probas
