"""
Isolation Forest anomaly scorer. Unsupervised — fit only on a reference
set of "normal" rows (currently preprocessing.select_spec_normal_rows(),
a spec-bound row filter; preprocessing.split_reference_window()'s naive
first-N-rows window still exists as a fallback path — see
train_isolation_forest.py / validate.py for which is actually wired up),
scored continuously on everything. Doesn't care whether that reference
set is a contiguous time window or scattered rows — fit()/score_samples()
treat it as an unordered set of feature vectors either way.

Outputs a raw anomaly score (sklearn's score_samples: higher = more
normal, lower = more anomalous) plus a normalized health percentage
calibrated against the reference set's own score distribution — see
health_from_score() and config.HEALTH_SENSITIVITY_STD.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

import config


class AnomalyScorer:
    def __init__(self):
        self.model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
        self.baseline_mean = None
        self.baseline_std = None
        self.baseline_feature_mean = None  # pd.Series, indexed by feature name
        self.baseline_feature_std = None   # pd.Series, indexed by feature name

    def fit(self, X_reference):
        """Fit on the reference/baseline window only, then calibrate the
        health-percentage mapping against that same window's own score
        distribution."""
        self.model.fit(X_reference)
        reference_scores = self.model.score_samples(X_reference)
        self.baseline_mean = float(np.mean(reference_scores))
        self.baseline_std = float(np.std(reference_scores)) or 1e-6  # avoid div-by-zero

        # Per-feature baseline, used only for diagnosis (which raw
        # feature(s) drove a given anomalous reading), never for the
        # score/health/status decision itself — that stays the single
        # combined score above.
        self.baseline_feature_mean = X_reference.mean(axis=0)
        self.baseline_feature_std = X_reference.std(axis=0).replace(0, np.nan)
        return self

    def score(self, X) -> np.ndarray:
        """Raw anomaly score — higher = more normal."""
        return self.model.score_samples(X)

    def feature_z_scores(self, X) -> "pd.Series":
        """
        Per-feature z-score of a single-row X against the reference
        window's per-feature mean/std: how many baseline standard
        deviations each individual feature is from "normal", independent
        of the others. Diagnostic only — use this to explain a WARN/
        CRITICAL reading (which feature(s), and by extension which
        sensor(s), moved the most), never to gate status on any one
        feature in isolation; the combined anomaly score already accounts
        for all features jointly and is what should drive the decision.
        """
        if self.baseline_feature_mean is None:
            raise RuntimeError("AnomalyScorer.fit() must be called before feature_z_scores().")
        row = X.iloc[0] if hasattr(X, "iloc") else pd.Series(X, index=self.baseline_feature_mean.index)
        z = (row - self.baseline_feature_mean) / self.baseline_feature_std
        return z.fillna(0.0)

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
        return {
            "baseline_mean": self.baseline_mean,
            "baseline_std": self.baseline_std,
            "baseline_feature_mean": self.baseline_feature_mean.to_dict(),
            "baseline_feature_std": self.baseline_feature_std.to_dict(),
        }

    @classmethod
    def from_calibration(cls, model, calibration: dict):
        obj = cls()
        obj.model = model
        obj.baseline_mean = calibration["baseline_mean"]
        obj.baseline_std = calibration["baseline_std"]
        # Older calibration.json files predate per-feature attribution —
        # fall back to None so feature_z_scores() fails loudly (via its
        # own check) rather than attribution silently reporting nothing.
        feat_mean = calibration.get("baseline_feature_mean")
        feat_std = calibration.get("baseline_feature_std")
        obj.baseline_feature_mean = pd.Series(feat_mean) if feat_mean is not None else None
        obj.baseline_feature_std = pd.Series(feat_std) if feat_std is not None else None
        return obj
