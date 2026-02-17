"""Feature and data distribution drift detection.

Monitors statistical properties of features/data over time
and raises alerts when distributions shift significantly.
"""

import numpy as np
from datetime import datetime
from typing import Optional

from loguru import logger

from core.types import Alert, AlertSeverity, AlertType


class DriftDetector:
    """Detects feature and data distribution drift using statistical tests.

    Methods:
      - Population Stability Index (PSI)
      - Kolmogorov-Smirnov test
      - Mean/variance shift detection
    """

    def __init__(
        self,
        psi_threshold: float = 0.2,
        ks_threshold: float = 0.05,
        mean_shift_std: float = 2.0,
        min_samples: int = 50,
    ) -> None:
        """Initialize drift detector.

        Args:
            psi_threshold: PSI above this → drift detected (0.1=moderate, 0.2=significant)
            ks_threshold: KS test p-value below this → drift detected
            mean_shift_std: Mean shift of more than this many std → drift
            min_samples: Minimum samples needed for meaningful comparison
        """
        self.psi_threshold = psi_threshold
        self.ks_threshold = ks_threshold
        self.mean_shift_std = mean_shift_std
        self.min_samples = min_samples

        self._reference_stats: dict[str, dict[str, float]] = {}
        self._reference_histograms: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def set_reference(self, feature_name: str, values: np.ndarray) -> None:
        """Set the reference distribution for a feature.

        Args:
            feature_name: Feature name
            values: Reference distribution values
        """
        clean = values[~np.isnan(values)]
        if len(clean) < self.min_samples:
            return

        self._reference_stats[feature_name] = {
            "mean": float(np.mean(clean)),
            "std": float(np.std(clean)),
            "median": float(np.median(clean)),
            "q25": float(np.percentile(clean, 25)),
            "q75": float(np.percentile(clean, 75)),
            "n": len(clean),
        }

        # Histogram for PSI
        hist, bin_edges = np.histogram(clean, bins=20, density=True)
        self._reference_histograms[feature_name] = (hist, bin_edges)

    def check_drift(
        self,
        feature_name: str,
        current_values: np.ndarray,
    ) -> Optional[Alert]:
        """Check if a feature has drifted from its reference distribution.

        Args:
            feature_name: Feature name
            current_values: Recent values to compare against reference

        Returns:
            Alert if drift detected, None otherwise
        """
        if feature_name not in self._reference_stats:
            return None

        clean = current_values[~np.isnan(current_values)]
        if len(clean) < self.min_samples:
            return None

        ref = self._reference_stats[feature_name]
        drift_signals = []

        # 1. Mean shift test
        current_mean = float(np.mean(clean))
        ref_std = max(ref["std"], 1e-10)
        mean_shift = abs(current_mean - ref["mean"]) / ref_std

        if mean_shift > self.mean_shift_std:
            drift_signals.append(
                f"mean_shift={mean_shift:.2f}σ (ref={ref['mean']:.4f}, cur={current_mean:.4f})"
            )

        # 2. Variance change
        current_std = float(np.std(clean))
        var_ratio = current_std / ref_std
        if var_ratio > 2.0 or var_ratio < 0.5:
            drift_signals.append(f"var_ratio={var_ratio:.2f}")

        # 3. PSI (Population Stability Index)
        psi = self._compute_psi(feature_name, clean)
        if psi is not None and psi > self.psi_threshold:
            drift_signals.append(f"PSI={psi:.4f}")

        # 4. KS test
        ks_pvalue = self._ks_test(feature_name, clean)
        if ks_pvalue is not None and ks_pvalue < self.ks_threshold:
            drift_signals.append(f"KS_pvalue={ks_pvalue:.4f}")

        if drift_signals:
            severity = AlertSeverity.CRITICAL if len(drift_signals) >= 3 else AlertSeverity.WARNING
            return Alert(
                alert_id=f"drift_{feature_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                severity=severity,
                alert_type=AlertType.DRIFT_DETECTED,
                message=f"Drift detected for {feature_name}: {'; '.join(drift_signals)}",
                timestamp=datetime.now(),
                payload={
                    "feature": feature_name,
                    "signals": drift_signals,
                    "psi": psi,
                    "mean_shift_sigma": mean_shift,
                },
            )

        return None

    def check_all_features(
        self,
        feature_dict: dict[str, np.ndarray],
    ) -> list[Alert]:
        """Check drift for all features.

        Args:
            feature_dict: {feature_name: current_values}

        Returns:
            List of drift alerts
        """
        alerts = []
        for name, values in feature_dict.items():
            alert = self.check_drift(name, values)
            if alert:
                alerts.append(alert)
                logger.warning(f"Drift: {alert.message}")

        return alerts

    def _compute_psi(self, feature_name: str, current: np.ndarray) -> Optional[float]:
        """Compute Population Stability Index."""
        if feature_name not in self._reference_histograms:
            return None

        ref_hist, bin_edges = self._reference_histograms[feature_name]
        cur_hist, _ = np.histogram(current, bins=bin_edges, density=True)

        # Add small epsilon to avoid log(0)
        eps = 1e-6
        ref_pct = ref_hist / max(ref_hist.sum(), eps) + eps
        cur_pct = cur_hist / max(cur_hist.sum(), eps) + eps

        psi = float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))
        return max(psi, 0)

    def _ks_test(self, feature_name: str, current: np.ndarray) -> Optional[float]:
        """Kolmogorov-Smirnov test against reference distribution."""
        try:
            from scipy.stats import kstest

            ref = self._reference_stats[feature_name]
            # Compare against normal with reference mean/std
            statistic, pvalue = kstest(
                current,
                'norm',
                args=(ref["mean"], max(ref["std"], 1e-10)),
            )
            return float(pvalue)
        except ImportError:
            return None
