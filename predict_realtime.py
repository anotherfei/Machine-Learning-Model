"""Reusable model-bundle and per-machine condition-monitoring engine.

The production PostgreSQL polling and persistence loop lives only in
worker.py. This module owns the shared inference pipeline used by that worker,
historical backfill, and accuracy simulation. It remains fully unsupervised:
health_status is never loaded, read, or referenced.
"""

from collections import deque

import numpy as np
import pandas as pd

import config
import artifact_utils
import feature_engineering
import machine_normalization
from kalman import HealthKalmanFilter
import trend_forecast
import failure_probability
import maintenance
import attribution
import runtime_config
from isolation_forest import AnomalyScorer

INFERENCE_BATCH_ROWS = 25_000


class ModelBundle:
    """One immutable shared-model bundle and its machine-specific context."""

    def __init__(self, base_dir: str | None = None, expected_version: str | None = None):
        self.scorer, self.feature_columns, self.metadata = artifact_utils.load_artifacts(base_dir)
        self.calibrations = artifact_utils.load_machine_calibrations(base_dir=base_dir) or {}
        self.normalizers = artifact_utils.load_machine_feature_normalizers(base_dir=base_dir) or {}
        self.version_id = self.metadata.get("version_id", self.metadata.get("trained_at", "unversioned"))
        if expected_version and self.version_id != expected_version:
            raise ValueError(
                f"Requested model {expected_version!r}, but bundle identifies as {self.version_id!r}"
            )

    def commissioned(self, machine_id: str) -> bool:
        machine_id = str(machine_id)
        return machine_id in self.calibrations and machine_id in self.normalizers

    def create_monitor(self, machine_id: str):
        machine_id = str(machine_id)
        calibration_item = self.calibrations.get(machine_id)
        normalizer = self.normalizers.get(machine_id)
        if not calibration_item or not normalizer:
            raise ValueError(
                f"Machine {machine_id!r} is not commissioned in model {self.version_id!r}."
            )
        local_scorer = AnomalyScorer.from_calibration(
            self.scorer.model, calibration_item["calibration"]
        )
        return SpindleMonitor(
            local_scorer,
            self.feature_columns,
            self.metadata,
            feature_normalizer_override=normalizer,
            machine_id=machine_id,
        )

class SpindleMonitor:
    """
    Three buffers, matched to the three timescales in play:

    - self.window: short rolling window (WINDOW_SIZE raw ticks) for
      feature engineering -> Isolation Forest score, same cadence as
      training.
    - self.kalman: single running filter, denoises the health-percentage
      signal tick by tick.
    - self.trend_history: longer lookback (TREND_LOOKBACK_MINUTES) of
      Kalman-smoothed health values, used only by trend_forecast.py.

    A fourth buffer, self.kalman_init_buffer, holds the first
    KALMAN_INIT_SAMPLES raw health readings so the Kalman filter can be
    seeded from their mean instead of a single noisy first tick (see
    config.KALMAN_INIT_SAMPLES). No result is returned while this buffer
    is filling, so the cold-start transient never reaches
    maintenance.recommend() as a false CRITICAL reading.
    """

    def __init__(self, scorer_override, feature_cols_override, metadata_override=None,
                 feature_normalizer_override=None, machine_id=None):
        if machine_id is None:
            raise ValueError("machine_id is required for a machine-scoped monitor")
        self.scorer = scorer_override
        self.feature_cols = feature_cols_override
        self.metadata = metadata_override or {}
        self.machine_id = str(machine_id)
        self.feature_normalizer = feature_normalizer_override
        self.window = deque(maxlen=config.WINDOW_SIZE)
        self.kalman = None
        self.kalman_init_samples = int(runtime_config.get(
            "KALMAN_INIT_SAMPLES", config.KALMAN_INIT_SAMPLES
        ))
        self.trend_lookback_minutes = int(runtime_config.get(
            "TREND_LOOKBACK_MINUTES", config.TREND_LOOKBACK_MINUTES
        ))
        self.kalman_init_buffer = deque(maxlen=self.kalman_init_samples)
        self.trend_window = trend_forecast.RollingTrendWindow(
            self.trend_lookback_minutes
        )
        # Backward-compatible aliases for diagnostics/runtime inspection.
        self.trend_minutes = self.trend_window.minutes
        self.trend_health = self.trend_window.health
        self._time_origin = None
        self._fallback_elapsed_minutes = -1.0
        self.tick = 0
        self.trend_fit_ticks = 0  # counts ticks where fit_trend() actually returned a fit
        self.maintenance = maintenance.MaintenanceDebouncer()

    def _elapsed_minutes(self, timestamp=None) -> float:
        """Return elapsed real minutes, with a one-minute fallback for legacy callers."""
        if timestamp is None:
            self._fallback_elapsed_minutes += 1.0
            return self._fallback_elapsed_minutes
        current = pd.Timestamp(timestamp)
        if self._time_origin is None:
            self._time_origin = current
        elapsed = (current - self._time_origin).total_seconds() / 60.0
        if elapsed < 0:
            raise ValueError("SpindleMonitor timestamps must be chronological")
        return float(elapsed)

    def update(self, reading: dict, timestamp=None) -> dict:
        """Process one live reading through the same batch-capable pipeline."""
        return self.update_many([reading], [timestamp])[0]

    def update_many(self, readings: list[dict], timestamps=None) -> list[dict | None]:
        """Process chronological readings with vectorized features and scoring.

        Stateful condition and maintenance stages are still evaluated one row
        at a time and in timestamp order.  Only the pure rolling-feature,
        normalization, and Isolation Forest stages are batched.  This keeps
        historical catch-up equivalent to live inference without paying the
        pandas/sklearn setup cost once per source row.
        """
        if not readings:
            return []
        if timestamps is None:
            timestamps = [None] * len(readings)
        if len(timestamps) != len(readings):
            raise ValueError("readings and timestamps must have the same length")

        elapsed_minutes = [self._elapsed_minutes(value) for value in timestamps]
        self.tick += len(readings)
        results = [None] * len(readings)

        # WINDOW_SIZE-1 rows are sufficient context for every feature belonging
        # to the new batch. self.window itself retains WINDOW_SIZE rows so the
        # public runtime state remains identical to one-at-a-time inference.
        context = list(self.window)[-max(0, config.WINDOW_SIZE - 1):]
        context_count = len(context)
        for reading in readings:
            self.window.append(reading)

        window_df = pd.DataFrame(context + readings)
        window_df[config.COL_TIMESTAMP] = pd.RangeIndex(len(window_df))
        feat_df = feature_engineering.create_features(window_df, verbose=False)
        if feat_df.empty:
            return results
        feat_df = feat_df[feat_df[config.COL_TIMESTAMP] >= context_count].copy()
        if feat_df.empty:
            return results

        raw_features = feat_df[self.feature_cols]
        normalized_features = machine_normalization.transform(
            raw_features,
            self.feature_cols,
            self.feature_normalizer,
            self.machine_id,
        )[self.feature_cols]
        raw_scores = self.scorer.score(normalized_features)
        health_values = self.scorer.health_from_score(raw_scores)

        for feature_number, source_position in enumerate(
            (feat_df[config.COL_TIMESTAMP].astype(int) - context_count).tolist()
        ):
            results[source_position] = self._finish_scored_reading(
                readings[source_position],
                elapsed_minutes[source_position],
                normalized_features.iloc[[feature_number]],
                raw_features.iloc[[feature_number]],
                float(raw_scores[feature_number]),
                float(health_values[feature_number]),
            )
        return results

    def _finish_scored_reading(
        self, reading, elapsed_minutes, latest_feats, latest_raw_feats,
        raw_score, health_raw,
    ):
        """Apply the necessarily sequential runtime stages to one score."""

        # ---- Kalman denoising ----
        if self.kalman is None:
            # Seed from the mean of the first few raw health readings
            # rather than a single tick, so the filter starts near the
            # true level instead of spending hours converging to it from
            # whatever the first noisy measurement happened to be.
            self.kalman_init_buffer.append(health_raw)
            if len(self.kalman_init_buffer) < self.kalman_init_samples:
                return None  # still warming up - emit nothing until seeded
            initial_level = float(np.mean(self.kalman_init_buffer))
            self.kalman = HealthKalmanFilter(initial_level=initial_level)
            health_state = self.kalman.level
        else:
            health_state = self.kalman.update(health_raw)

        # ---- Trend forecasting ----
        self.trend_window.append(elapsed_minutes, health_state)
        fit = self.trend_window.fit()

        if fit is None:
            remaining = config.REMAINING_DAYS_CAP
            prob_table = {h: 0.0 for h in config.FAILURE_PROB_HORIZONS_DAYS}
            slope, residual_std = 0.0, 0.0
            trend_trusted = True  # nothing to distrust - these are inert placeholders
        else:
            self.trend_fit_ticks += 1
            settled = self.trend_fit_ticks >= runtime_config.get(
                "TREND_SETTLE_TICKS", config.TREND_SETTLE_TICKS
            )
            slope, intercept, residual_std = fit
            significant = self.trend_window.slope_is_significant(
                slope, residual_std
            )
            trend_trusted = settled and significant
            remaining = trend_forecast.remaining_days(elapsed_minutes, health_state, slope)
            trend_minutes, trend_health = self.trend_window.arrays()
            diffusion = trend_forecast.estimate_diffusion(
                trend_minutes, trend_health, slope, intercept
            )
            prob_table = failure_probability.failure_probability_table(
                health_state, slope, diffusion,
            )

        rec = self.maintenance.evaluate(
            health_state,
            remaining,
            prob_table,
            trend_trusted=trend_trusted,
            elapsed_minutes=elapsed_minutes,
        )

        # ---- Diagnosis-only: which sensor(s) drove this reading ----
        # Computed every tick (cheap - a handful of subtractions over the
        # feature vector) but only attached to the result when something
        # actually fired, so a healthy run's output doesn't get noisier
        # for no reason. Never feeds back into rec/health_state/status -
        # those are already decided above from the single combined score.
        top_contributors = None
        if rec["level"] != "OK":
            z_scores = self.scorer.feature_z_scores(latest_feats)
            top_contributors = attribution.top_contributors(z_scores, top_k=3)

        # Keep the inference contract portable across PostgreSQL, JSON, the
        # WebSocket API, backfill, and historical simulation. NumPy's round()
        # can preserve np.float64, so normalize recursively before any caller
        # receives the result.
        return artifact_utils.to_json_safe({
            **{col: reading[col] for col in config.RAW_SENSOR_COLS},
            "anomaly_score": round(float(raw_score), 5),
            "health_raw": round(health_raw, 2),
            "health_state": round(health_state, 2),
            "trend_slope_per_day": round(slope * 24 * 60, 4),
            "remaining_days": remaining,
            "failure_probability": prob_table,
            "maintenance": rec,
            "top_contributors": top_contributors,
            # Persist physical/engineered values. A future shadow model applies
            # its own machine normalizer instead of inheriting today's scale.
            "feature_vector": {k: float(v) for k, v in latest_raw_feats.iloc[0].items()},
            "model_version": self.metadata.get("version_id", self.metadata.get("trained_at", "unversioned")),
        })
