"""
Isolation Forest anomaly scorer. Unsupervised — fit only on the reference
baseline window (see config.REFERENCE_WINDOW_MINUTES / preprocessing.
split_reference_window), scored continuously on everything.

Outputs a raw anomaly score (sklearn's score_samples: higher = more
normal, lower = more anomalous) plus a normalized health percentage
calibrated against the reference window's own score distribution — see
health_from_score() and config.HEALTH_SENSITIVITY_STD.
"""

import numpy as np
from sklearn.ensemble import IsolationForest

import config


class AnomalyScorer:
    def __init__(self):
        self.model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
        self.baseline_mean = None
        self.baseline_std = None

    def fit(self, X_reference):
        """Fit on the reference/baseline window only, then calibrate the
        health-percentage mapping against that same window's own score
        distribution."""
        self.model.fit(X_reference)
        reference_scores = self.model.score_samples(X_reference)
        self.baseline_mean = float(np.mean(reference_scores))
        self.baseline_std = float(np.std(reference_scores)) or 1e-6  # avoid div-by-zero
        return self

    def score(self, X) -> np.ndarray:
        """Raw anomaly score — higher = more normal."""
        return self.model.score_samples(X)

    def health_from_score(self, scores) -> np.ndarray:
        """
        Maps raw anomaly score -> health percentage [0, 100], calibrated
        against the reference window: 100 at or above the baseline mean,
        linearly down to 0 at HEALTH_SENSITIVITY_STD standard deviations
        below it. This mapping is fixed at fit() time from the baseline
        window only — it does not adapt as new (possibly degraded) data
        arrives, which is the correct behavior: health should read low
        once the asset is actually behaving very differently from its
        reference period, not renormalize back to "normal" as more
        degraded data accumulates.
        """
        if self.baseline_mean is None:
            raise RuntimeError("AnomalyScorer.fit() must be called before health_from_score().")

        z = (self.baseline_mean - np.asarray(scores)) / (config.HEALTH_SENSITIVITY_STD * self.baseline_std)
        health = 100.0 * (1.0 - np.clip(z, 0.0, 1.0))
        return health

    def calibration(self) -> dict:
        return {"baseline_mean": self.baseline_mean, "baseline_std": self.baseline_std}

    @classmethod
    def from_calibration(cls, model, calibration: dict):
        obj = cls()
        obj.model = model
        obj.baseline_mean = calibration["baseline_mean"]
        obj.baseline_std = calibration["baseline_std"]
        return obj
