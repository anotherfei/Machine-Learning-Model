"""
Global configuration for the spindle condition-monitoring pipeline.

Architecture (see README.md for the full walkthrough of why):

    Sensors -> Feature Engineering -> Isolation Forest -> Anomaly Score
        -> Kalman Filter -> Estimated Health State -> Trend Forecasting
        -> Remaining Useful Life -> Failure Probability
        -> Maintenance Recommendation

This is fully UNSUPERVISED. The raw CSV has a `health_status` column
(normal/warning/critical) but it is intentionally never loaded, read, or
referenced anywhere in this codebase — preprocessing.py selects columns
by name at read time specifically to exclude it structurally, not just by
convention. It exists only for the person running this pipeline to
manually spot-check results against, entirely outside this code. Do not
reintroduce it into any module — that would silently turn this back into
the label-dependent pipeline this one replaced.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "spindle.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed.csv")
FEATURES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "features.csv")

# ---------------------------------------------------------------------------
# Raw column names
# ---------------------------------------------------------------------------
COL_TIMESTAMP = "timestamp"
COL_VIBRATION = "vibration_mps2"
COL_CURRENT = "current_ampere"
COL_TEMPERATURE = "temperature_c"

RAW_SENSOR_COLS = [COL_VIBRATION, COL_CURRENT, COL_TEMPERATURE]

# ---------------------------------------------------------------------------
# Spec-based "normal operation" bounds (alternative to a time-window-based
# reference period — see preprocessing.select_spec_normal_rows()).
# PLACEHOLDER VALUES: these are Predictive_Maintenance.zip's spec-shaped
# filter (vibration<=3.0, temp<=50, current<=6.0), carried over because
# they were the only bounds available at the time this was written. This
# only produces a correct reference set if these are genuinely the
# manufacturer's rated normal range for this spindle. If they're a rough
# guess, that error is silent — select_spec_normal_rows() has no way to
# detect a wrong bound, only a missing one. Confirm against the actual
# spec sheet before relying on this in production.
SPEC_VIBRATION_MAX = 3.0
SPEC_TEMPERATURE_MAX = 50.0
SPEC_CURRENT_MAX = 6.0
# NOTE: deliberately no COL_STATUS here — see module docstring.

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
WINDOW_SIZE = 10          # rolling window, in rows (1 row = 1 minute in this data)
SAMPLING_RATE_HZ = 1 / 60
MIN_PERIODS = WINDOW_SIZE

FEATURE_CONFIG = {
    "window_size": WINDOW_SIZE,
    "min_periods": MIN_PERIODS,
    "sampling_rate_hz": SAMPLING_RATE_HZ,
    "sensor_cols": RAW_SENSOR_COLS,
}

# ---------------------------------------------------------------------------
# Reference / commissioning baseline window
# ---------------------------------------------------------------------------
# Isolation Forest is unsupervised but still needs SOMETHING to define
# "normal" relative to. Standard industrial practice: use an early
# commissioning/burn-in period as the reference baseline, on the
# engineering assumption that a freshly commissioned or recently serviced
# asset starts in good condition. This is an operational assumption, NOT
# derived from any label in the data — if the assumption is wrong (the
# asset was already degrading during this window), the whole pipeline's
# calibration is off. That's a real limitation of unsupervised monitoring
# in general, not specific to this dataset — flag it, don't hide it.
REFERENCE_WINDOW_MINUTES = 2 * 24 * 60  # first 2 days

# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------
ISOLATION_FOREST_PARAMS = {
    "n_estimators": 200,
    "max_samples": "auto",
    "contamination": "auto",
    "random_state": 42,
}

# ---------------------------------------------------------------------------
# Anomaly score -> health percentage mapping
# ---------------------------------------------------------------------------
# health = 100 when score is at or above the reference window's mean
# (as normal as the baseline), degrading linearly to 0 at
# HEALTH_SENSITIVITY_STD standard deviations below the baseline mean.
# Tune this if health hits 0% too early/late relative to visible wear.
HEALTH_SENSITIVITY_STD = 4.0

# ---------------------------------------------------------------------------
# Kalman filter — denoises the raw health-percentage signal into a
# smoothed "estimated health state". Deliberately a simple constant-level
# (random-walk) filter with NO velocity/trend state — trend estimation is
# a separate, swappable stage (trend_forecast.py), not folded into the
# Kalman filter. Keeping these responsibilities separate is the whole
# point of this architecture (see README).
# ---------------------------------------------------------------------------
KALMAN_PARAMS = {
    "process_var": 0.01,     # how much true health can drift per tick
    "measurement_var": 9.0,  # noise in the raw health-percentage estimate
}

# ---------------------------------------------------------------------------
# Trend forecasting
# ---------------------------------------------------------------------------
# Linear regression over the last TREND_LOOKBACK_MINUTES of Kalman-smoothed
# health values. Long enough to average out noise, short enough to react
# if the degradation rate changes.
TREND_LOOKBACK_MINUTES = 720  # 12 hours — 4 hours was noisy enough to cause
                                # false URGENT flags during flat, healthy periods;
                                # see README for the specific example found
TREND_MIN_POINTS = 60

FAILURE_HEALTH_THRESHOLD = 20  # health % at which the asset is considered failed
REMAINING_DAYS_CAP = 90

# ---------------------------------------------------------------------------
# Failure probability
# ---------------------------------------------------------------------------
# Degradation modeled as a random walk with drift (standard assumption in
# RUL literature): forecast uncertainty grows with sqrt(horizon), using
# the trend fit's residual std as the per-step noise estimate.
#
# Horizons restricted to <=1 day. validate.py's calibration check measured
# actual Brier scores against health_status: 0.5d=0.164, 1d=0.215 (both
# beat the uninformative baseline of 0.25) but 2d=0.311, 3d=0.325 (WORSE
# than guessing 50/50 — not just unproven, actively misleading). An
# earlier version of this list went out to 35 days, copied from an
# illustrative example without checking it against this dataset's own
# ~6.9-day span or measured calibration — don't extend this list past 1
# day without rerunning validate.py and confirming the Brier score still
# beats 0.25 at whatever horizon you add.
FAILURE_PROB_HORIZONS_DAYS = [0.25, 0.5, 0.75, 1]

# ---------------------------------------------------------------------------
# Maintenance recommendation rules
# ---------------------------------------------------------------------------
MAINTENANCE_HORIZON_DAYS = 1          # "how soon" horizon the rules check against — kept within the <=1 day range validated above
MAINTENANCE_PROB_URGENT = 0.70       # failure probability within horizon -> urgent
MAINTENANCE_PROB_PLAN = 0.30         # -> plan maintenance
MAINTENANCE_REMAINING_DAYS_URGENT = 7
MAINTENANCE_HEALTH_INSPECT = 40      # health % below this -> inspect regardless