"""
Real-time condition-monitoring loop — replaces Predictive_Maintenance.zip's
main.py / predict_machine() entirely, and replaces the intermediate
LightGBM+Kalman version of this repo. Full pipeline, per README.md:

    Sensors -> Feature Engineering -> Isolation Forest -> Anomaly Score
        -> Kalman Filter -> Estimated Health State -> Trend Forecasting
        -> Remaining Useful Life -> Failure Probability
        -> Maintenance Recommendation

Fully unsupervised — health_status is never loaded, read, or referenced
anywhere in this script or anything it imports (see config.py docstring).

As with the earlier version: no live hardware exists in this environment,
so read_sensor() replays config.RAW_DATA_PATH row by row instead of
generating random numbers. Point this at a live feed by replacing
read_sensor()'s body — everything downstream is unchanged.

Usage:
    python predict_realtime.py
"""

import time
from collections import deque

import os
import numpy as np
import pandas as pd

import config
import artifact_utils
import feature_engineering
from kalman import HealthKalmanFilter
import trend_forecast
import failure_probability
import maintenance
import attribution


print("Loading model...")
scorer, feature_cols, metadata = artifact_utils.load_artifacts()
print(f"Model loaded ({metadata['n_features']} features, trained {metadata['trained_at']}).")


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

    def __init__(self):
        self.window = deque(maxlen=config.WINDOW_SIZE)
        self.kalman = None
        self.kalman_init_buffer = deque(maxlen=config.KALMAN_INIT_SAMPLES)
        self.trend_minutes = deque(maxlen=config.TREND_LOOKBACK_MINUTES)
        self.trend_health = deque(maxlen=config.TREND_LOOKBACK_MINUTES)
        self.tick = 0

    def update(self, reading: dict) -> dict:
        self.window.append(reading)
        self.tick += 1

        if len(self.window) < config.WINDOW_SIZE:
            return None  # not enough history for rolling features yet

        # ---- Feature engineering + Isolation Forest ----
        window_df = pd.DataFrame(self.window)
        window_df[config.COL_TIMESTAMP] = pd.RangeIndex(len(window_df))  # dummy, unused by features

        feat_df = feature_engineering.create_features(window_df, verbose=False)
        if feat_df.empty:
            return None
        latest_feats = feat_df.iloc[[-1]][feature_cols]

        raw_score = scorer.score(latest_feats)[0]
        health_raw = float(scorer.health_from_score(np.array([raw_score]))[0])

        # ---- Kalman denoising ----
        if self.kalman is None:
            # Seed from the mean of the first few raw health readings
            # rather than a single tick, so the filter starts near the
            # true level instead of spending hours converging to it from
            # whatever the first noisy measurement happened to be.
            self.kalman_init_buffer.append(health_raw)
            if len(self.kalman_init_buffer) < config.KALMAN_INIT_SAMPLES:
                return None  # still warming up — emit nothing until seeded
            initial_level = float(np.mean(self.kalman_init_buffer))
            self.kalman = HealthKalmanFilter(initial_level=initial_level)
            health_state = self.kalman.level
        else:
            health_state = self.kalman.update(health_raw)

        # ---- Trend forecasting ----
        self.trend_minutes.append(self.tick)
        self.trend_health.append(health_state)
        fit = trend_forecast.fit_trend(np.array(self.trend_minutes), np.array(self.trend_health))

        if fit is None:
            remaining = config.REMAINING_DAYS_CAP
            prob_table = {h: 0.0 for h in config.FAILURE_PROB_HORIZONS_DAYS}
            slope, residual_std = 0.0, 0.0
        else:
            slope, intercept, residual_std = fit
            remaining = trend_forecast.remaining_days(self.tick, health_state, slope)
            prob_table = failure_probability.failure_probability_table(health_state, slope, residual_std)

        rec = maintenance.recommend(health_state, remaining, prob_table)

        # ---- Diagnosis-only: which sensor(s) drove this reading ----
        # Computed every tick (cheap — a handful of subtractions over the
        # feature vector) but only attached to the result when something
        # actually fired, so a healthy run's output doesn't get noisier
        # for no reason. Never feeds back into rec/health_state/status —
        # those are already decided above from the single combined score.
        top_contributors = None
        if rec["level"] != "OK":
            z_scores = scorer.feature_z_scores(latest_feats)
            top_contributors = attribution.top_contributors(z_scores, top_k=3)

        return {
            "vibration": reading[config.COL_VIBRATION],
            "temperature": reading[config.COL_TEMPERATURE],
            "current": reading[config.COL_CURRENT],
            "anomaly_score": round(float(raw_score), 5),
            "health_raw": round(health_raw, 2),
            "health_state": round(health_state, 2),
            "trend_slope_per_day": round(slope * 24 * 60, 4),
            "remaining_days": remaining,
            "failure_probability": prob_table,
            "maintenance": rec,
            "top_contributors": top_contributors,
        }


def read_sensor():
    df = pd.read_csv(config.RAW_DATA_PATH, usecols=[config.COL_TIMESTAMP] + config.RAW_SENSOR_COLS)
    for _, row in df.iterrows():
        yield {
            config.COL_VIBRATION: float(row[config.COL_VIBRATION]),
            config.COL_TEMPERATURE: float(row[config.COL_TEMPERATURE]),
            config.COL_CURRENT: float(row[config.COL_CURRENT]),
        }


LOG_PATH = config.REALTIME_PREDICTIONS_PATH


def save_result(result: dict):
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    flat = {k: v for k, v in result.items()
            if k not in ("failure_probability", "maintenance", "top_contributors")}
    flat["maintenance_level"] = result["maintenance"]["level"]
    flat["maintenance_reason"] = result["maintenance"]["reason"]
    for h, p in result["failure_probability"].items():
        flat[f"fail_prob_{h}d"] = p
    flat["top_contributors"] = (
        "; ".join(f"{name} (z={z:+.1f})" for name, z, _ in result["top_contributors"])
        if result["top_contributors"] else ""
    )
    flat["timestamp"] = pd.Timestamp.now()

    row = pd.DataFrame([flat])
    try:
        old = pd.read_csv(LOG_PATH)
        pd.concat([old, row], ignore_index=True).to_csv(LOG_PATH, index=False)
    except FileNotFoundError:
        row.to_csv(LOG_PATH, index=False)


if __name__ == "__main__":
    monitor = SpindleMonitor()
    print("Realtime Spindle Monitoring Started (replaying logged data)")
    tick = 0

    for reading in read_sensor():
        try:
            result = monitor.update(reading)
            if result is None:
                continue

            tick += 1
            print("\n" + "=" * 60)
            print(f"data ke :{tick}")
            print("Vibration :", result["vibration"])
            print("Temperature :", result["temperature"])
            print("Current :", result["current"])
            print("Anomaly Score :", result["anomaly_score"])
            print("Health (raw) :", result["health_raw"], "%")
            print("Health (Kalman-smoothed) :", result["health_state"], "%")
            print("Trend :", result["trend_slope_per_day"], "%/day")
            print("Remaining Days :", result["remaining_days"])
            print("Failure Probability :", result["failure_probability"])
            print("Maintenance :", result["maintenance"]["level"], "-", result["maintenance"]["reason"])
            if result["top_contributors"]:
                names = ", ".join(f"{name} (z={z:+.1f})" for name, z, _ in result["top_contributors"])
                print("Top contributors :", names)

            save_result(result)
            time.sleep(0.01)  # replaying logged data — not throttled to real time

        except Exception as e:
            print("ERROR :", e)
            time.sleep(1)
