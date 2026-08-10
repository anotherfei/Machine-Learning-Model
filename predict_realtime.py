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

read_sensor() polls a Postgres table (see db.py) every
POLL_INTERVAL_SECONDS for rows newer than the last one it has already
processed, and yields them in order. Connection info + table name come
from env vars (PG_HOST/PG_PORT/PG_DATABASE/PG_USER/PG_PASSWORD/PG_TABLE
— see .env.example), with optional --host/--port/--db/--user/--password/
--table CLI overrides. Runs forever; stop with Ctrl+C.

Usage:
    python predict_realtime.py
    python predict_realtime.py --host ... --user ... --password ...
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
import db

POLL_INTERVAL_SECONDS = 60


scorer = None
feature_cols = None
metadata = None

def reload_model():
    """Reload the active on-disk artifact bundle between ticks."""
    global scorer, feature_cols, metadata
    scorer, feature_cols, metadata = artifact_utils.load_artifacts()
    return scorer, feature_cols, metadata


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

    def __init__(self, scorer_override=None, feature_cols_override=None, metadata_override=None):
        global scorer, feature_cols, metadata
        if scorer_override is not None:
            self.scorer = scorer_override
            self.feature_cols = feature_cols_override
            self.metadata = metadata_override or {}
        else:
            if scorer is None:
                reload_model()
            self.scorer = scorer
            self.feature_cols = feature_cols
            self.metadata = metadata or {}
        self.window = deque(maxlen=config.WINDOW_SIZE)
        self.kalman = None
        self.kalman_init_buffer = deque(maxlen=config.KALMAN_INIT_SAMPLES)
        self.trend_minutes = deque(maxlen=config.TREND_LOOKBACK_MINUTES)
        self.trend_health = deque(maxlen=config.TREND_LOOKBACK_MINUTES)
        self.tick = 0
        self.trend_fit_ticks = 0  # counts ticks where fit_trend() actually returned a fit
        self.maintenance = maintenance.MaintenanceDebouncer()

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
        latest_feats = feat_df.iloc[[-1]][self.feature_cols]

        raw_score = self.scorer.score(latest_feats)[0]
        health_raw = float(self.scorer.health_from_score(np.array([raw_score]))[0])

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
            trend_trusted = True  # nothing to distrust — these are inert placeholders
        else:
            self.trend_fit_ticks += 1
            settled = self.trend_fit_ticks >= config.TREND_SETTLE_TICKS
            slope, intercept, residual_std = fit
            significant = trend_forecast.slope_is_significant(
                np.array(self.trend_minutes), slope, residual_std)
            trend_trusted = settled and significant
            remaining = trend_forecast.remaining_days(self.tick, health_state, slope)
            prob_table = failure_probability.failure_probability_table(health_state, slope, residual_std)

        rec = self.maintenance.evaluate(health_state, remaining, prob_table, trend_trusted=trend_trusted)

        # ---- Diagnosis-only: which sensor(s) drove this reading ----
        # Computed every tick (cheap — a handful of subtractions over the
        # feature vector) but only attached to the result when something
        # actually fired, so a healthy run's output doesn't get noisier
        # for no reason. Never feeds back into rec/health_state/status —
        # those are already decided above from the single combined score.
        top_contributors = None
        if rec["level"] != "OK":
            z_scores = self.scorer.feature_z_scores(latest_feats)
            top_contributors = attribution.top_contributors(z_scores, top_k=3)

        return {
            **{col: reading[col] for col in config.RAW_SENSOR_COLS},
            "anomaly_score": round(float(raw_score), 5),
            "health_raw": round(health_raw, 2),
            "health_state": round(health_state, 2),
            "trend_slope_per_day": round(slope * 24 * 60, 4),
            "remaining_days": remaining,
            "failure_probability": prob_table,
            "maintenance": rec,
            "top_contributors": top_contributors,
            "feature_vector": {k: float(v) for k, v in latest_feats.iloc[0].items()},
            "model_version": self.metadata.get("version_id", self.metadata.get("trained_at", "unversioned")),
        }


def read_sensor():
    """
    Polls Postgres every POLL_INTERVAL_SECONDS for rows newer than the
    last one already yielded (watermark = last row's timestamp), yielding
    them in ascending timestamp order — same in-order, gapless contract
    SpindleMonitor's rolling window / Kalman filter relied on when this
    read from a CSV. Runs forever; there is no natural end to a live feed.

    First poll (last_seen=None) pulls everything currently in the table,
    same as the old full-CSV replay. After that, each poll only pulls
    what's new since the previous one.
    """
    args = db.parse_args()
    conn = db.get_connection(args)
    table = db.get_table_name(args)
    dbcols = db.get_db_columns()
    last_seen = None

    while True:
        rows = db.fetch_new_rows(conn, table, since=last_seen)
        for row in rows:
            yield {
                **{col: float(row[dbcols["by_config_name"][col]]) for col in config.RAW_SENSOR_COLS},
                "_timestamp": row[dbcols["timestamp"]],
            }
            last_seen = row[dbcols["timestamp"]]
        time.sleep(POLL_INTERVAL_SECONDS)


LOG_PATH = config.REALTIME_PREDICTIONS_PATH


def save_result(result: dict, reset: bool = False):
    """
    reset=True truncates LOG_PATH before writing — used once at the start
    of a run so each run's log corresponds to exactly one replay of
    config.PREDICT_DATA_PATH. Without this, every run appended onto
    whatever was already there (confirmed: results/realtime_predictions.csv
    had grown to 11,842 rows despite PREDICT_DATA_PATH having only 10,000 —
    leftover rows from a previous run/dataset silently mixed in with no way
    to tell which prediction came from which run).

    Appends directly rather than re-reading the whole file every tick
    (the previous approach was O(n^2) over a run) — writes the header
    only on the first row of a run.
    """
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    flat = {k: v for k, v in result.items()
            if k not in ("failure_probability", "maintenance", "top_contributors", "feature_vector")}
    flat["maintenance_level"] = result["maintenance"]["level"]
    flat["maintenance_reason"] = result["maintenance"]["reason"]
    flat["maintenance_trigger"] = result["maintenance"].get("trigger", "none")
    for h, p in result["failure_probability"].items():
        flat[f"fail_prob_{h}d"] = p
    flat["top_contributors"] = (
        "; ".join(f"{name} (z={z:+.1f})" for name, z, _ in result["top_contributors"])
        if result["top_contributors"] else ""
    )
    flat["timestamp"] = result.get("_timestamp", pd.Timestamp.now())

    row = pd.DataFrame([flat])
    write_header = reset or not os.path.exists(LOG_PATH)
    row.to_csv(LOG_PATH, mode="w" if reset else "a", header=write_header, index=False)


if __name__ == "__main__":
    monitor = SpindleMonitor()
    print("Realtime Spindle Monitoring Started (polling Postgres)")
    print(f"Polling every {POLL_INTERVAL_SECONDS}s")
    tick = 0
    log_reset_done = False

    for reading in read_sensor():
        try:
            result = monitor.update(reading)
            if result is None:
                continue

            tick += 1
            print("\n" + "=" * 60)
            print(f"data ke : {tick}")
            for col in config.RAW_SENSOR_COLS:
                print(f"{col} :", result[col])
            print("Anomaly Score :", result["anomaly_score"])
            print("Health (raw) :", result["health_raw"], "%")
            print("Health (Kalman-smoothed) :", result["health_state"], "%")
            print("Trend :", result["trend_slope_per_day"], "% / day")
            print("Remaining Days :", result["remaining_days"])
            print("Failure Probability :", result["failure_probability"])
            print("Maintenance :", result["maintenance"]["level"], "-", result["maintenance"]["reason"])
            if result["top_contributors"]:
                names = ", ".join(f"{name} (z={z:+.1f})" for name, z, _ in result["top_contributors"])
                print("Top contributors :", names)

            save_result(result, reset=not log_reset_done)
            log_reset_done = True

        except Exception as e:
            print("ERROR :", e)
            time.sleep(1)
