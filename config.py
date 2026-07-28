"""
Global configuration for the Gearbox RUL pipeline.
Single source of truth for paths, feature engineering params, and model hyperparameters.
Nothing in this project should hardcode a value that lives here.
"""

import os

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DATA_DIR = os.path.join(ROOT_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(ROOT_DIR, "data", "processed")
ARTIFACTS_DIR = os.path.join(ROOT_DIR, "artifacts")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "raw.csv")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "processed.csv")
FEATURES_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "features.csv")

# ---------------------------------------------------------------------------
# Raw column names — adjust to match your actual CSV headers
# ---------------------------------------------------------------------------
COL_UNIT_ID = "unit_id"        # which gearbox this row belongs to
COL_TIMESTAMP = "timestamp"
COL_VIBRATION = "vibration"
COL_CURRENT = "current"
COL_TEMPERATURE = "temperature"
COL_RUL = "RUL"

RAW_SENSOR_COLS = [COL_VIBRATION, COL_CURRENT, COL_TEMPERATURE]

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
# Rolling window size, in ROWS (not seconds) — convert from your actual
# sampling rate before setting this. E.g. 30 rows at 1 Hz = 30 seconds.
WINDOW_SIZE = 30
SAMPLING_RATE_HZ = 1  # informational — stored in metadata, used for docs/plots only

# Minimum number of valid rows required before a window produces a feature
# value instead of NaN. Setting this equal to WINDOW_SIZE means "no partial
# windows allowed" — the safest default; loosen deliberately, not by accident.
MIN_PERIODS = WINDOW_SIZE

# RUL labeling
RUL_CAP = 125  # piecewise-linear cap: RUL flat at this value until degradation onset
# If your CSV already contains a ground-truth RUL column, PIECEWISE_LABELING=False
# and this cap is ignored — set it only if you need to derive RUL from failure timestamps.
PIECEWISE_LABELING = True

# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
N_SPLITS = 5          # GroupKFold folds, grouped by unit_id
TEST_UNIT_FRACTION = 0.2  # used only by the simple holdout split in preprocessing.py
RANDOM_STATE = 42

# ---------------------------------------------------------------------------
# Feature engineering config dict — hashed and stored in metadata.json so
# predict.py can detect drift between the config used at train time vs.
# whatever config.py contains at inference time.
# ---------------------------------------------------------------------------
FEATURE_CONFIG = {
    "window_size": WINDOW_SIZE,
    "min_periods": MIN_PERIODS,
    "sampling_rate_hz": SAMPLING_RATE_HZ,
    "sensor_cols": RAW_SENSOR_COLS,
}

# ---------------------------------------------------------------------------
# Model hyperparameters — kept per-model so each models/*.py stays thin
# ---------------------------------------------------------------------------
LINEAR_REGRESSION_PARAMS = {}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 300,
    "max_depth": 12,
    "min_samples_leaf": 5,
    "n_jobs": -1,
    "random_state": RANDOM_STATE,
}

SVR_PARAMS = {
    "kernel": "rbf",
    "C": 10.0,
    "epsilon": 0.5,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": -1,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
}

# ---------------------------------------------------------------------------
# PHM08 asymmetric scoring function parameters
# ---------------------------------------------------------------------------
PHM08_ALPHA_EARLY = 13   # penalty denominator when prediction is early (pred < true)
PHM08_ALPHA_LATE = 10    # penalty denominator when prediction is late (pred > true) — steeper
