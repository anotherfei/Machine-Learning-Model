"""
Loading and cleaning for the spindle condition-monitoring pipeline.

CRITICAL: load_data() reads ONLY [timestamp] + RAW_SENSOR_COLS via
pandas' `usecols`. This is deliberate and structural, not just a
convention to remember — health_status exists in the raw CSV but must
never enter this codebase (see config.py's module docstring for why).
Using `usecols` means even a future column-selection bug elsewhere can't
accidentally pull it in; it's dropped before anything else runs.
"""

import os
import hashlib
import json

import pandas as pd

import config


def raw_data_signature(path: str = None) -> str:
    """
    Fingerprint of a raw data file (path + size + mtime), used to detect
    when config.RAW_DATA_PATH has been repointed at a different file (or
    the same file edited in place) since data/processed/*.csv was last
    built. Content isn't hashed — path+size+mtime is enough to catch a
    changed source and is far cheaper on a file this size, checked on
    every run.
    """
    path = path or config.RAW_DATA_PATH
    stat = os.stat(path)
    fingerprint = f"{os.path.abspath(path)}:{stat.st_size}:{stat.st_mtime}"
    return hashlib.md5(fingerprint.encode()).hexdigest()


def save_source_signature(path: str = None):
    sig = raw_data_signature(path)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    with open(os.path.join(config.PROCESSED_DATA_DIR, "source_signature.json"), "w") as f:
        json.dump({"source_path": path or config.RAW_DATA_PATH, "signature": sig}, f, indent=2)


def cached_features_are_stale(path: str = None) -> bool:
    """
    True if data/processed/features.csv either doesn't exist, has no
    recorded source signature, or was built from a different raw file
    than config.RAW_DATA_PATH currently points to. See
    train_isolation_forest.py's get_or_build_features() — this is what
    stops it from silently training on stale cached features after
    RAW_DATA_PATH changes, which happened for real: switching to
    spindle_train.csv left features.csv (9,992 rows, the old ~10k-row
    file) untouched and get_or_build_features() had no way to notice.
    """
    if not os.path.exists(config.FEATURES_DATA_PATH):
        return True
    sig_path = os.path.join(config.PROCESSED_DATA_DIR, "source_signature.json")
    if not os.path.exists(sig_path):
        return True
    with open(sig_path) as f:
        saved = json.load(f)
    return saved.get("signature") != raw_data_signature(path)


def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    usecols = [config.COL_TIMESTAMP] + config.RAW_SENSOR_COLS
    df = pd.read_csv(path, usecols=usecols)

    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw CSV is missing required columns: {missing}. "
            f"Check config.py column names against your actual CSV headers."
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP], errors="coerce")
    df = df.sort_values(config.COL_TIMESTAMP)

    before = len(df)
    df = df.drop_duplicates(subset=[config.COL_TIMESTAMP])
    dropped = before - len(df)
    if dropped:
        print(f"[clean_data] Dropped {dropped} duplicate timestamp rows.")

    before = len(df)
    df = df.dropna(subset=config.RAW_SENSOR_COLS)
    dropped = before - len(df)
    if dropped:
        print(f"[clean_data] Dropped {dropped} rows with missing sensor values.")

    # Remove physically impossible readings — adjust bounds to your sensor specs
    df = df[
        (df[config.COL_VIBRATION] >= 0)
        & (df[config.COL_CURRENT] >= 0)
        & (df[config.COL_TEMPERATURE] > -50)
        & (df[config.COL_TEMPERATURE] < 300)
    ]

    df = df.reset_index(drop=True)
    return df


def split_reference_window(df: pd.DataFrame):
    """
    Splits off the commissioning/burn-in baseline window used to fit the
    Isolation Forest and calibrate the anomaly-score-to-health mapping.
    See config.REFERENCE_WINDOW_MINUTES docstring for the assumption this
    rests on. Purely a row-count cutoff on the chronologically sorted
    data — not derived from any label.
    """
    df = df.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
    cutoff = min(config.REFERENCE_WINDOW_MINUTES, len(df))

    reference_df = df.iloc[:cutoff].reset_index(drop=True)
    rest_df = df.iloc[cutoff:].reset_index(drop=True)

    print(
        f"[split_reference_window] Reference window: {len(reference_df)} rows "
        f"({reference_df[config.COL_TIMESTAMP].min()} -> {reference_df[config.COL_TIMESTAMP].max()})"
    )
    return reference_df, rest_df


def select_spec_normal_rows(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Row-wise "normal operation" filter, checked against fixed rated
    bounds (config.SPEC_*) instead of a contiguous burn-in time window.
    Alternative to split_reference_window() for defining what the
    Isolation Forest gets fit on.

    Applied to raw sensor readings (config.RAW_SENSOR_COLS) row-by-row —
    NOT to rolling features. A spec bound is a claim about a single
    instantaneous reading ("vibration above 3.0 is outside rated
    operation"); checking it against a rolling mean would average out
    exactly the transient excursions this filter exists to catch, and
    would silently let a spiking-but-averaged-out row through.

    Unlike a time-window burn-in period, this doesn't require an early
    "clean" stretch to exist in the data at all — every row is judged on
    its own, independent of when it occurred. That also means it's not
    anchored to trajectory shape (works the same on a single ramp-to-
    failure or a cyclic pattern). The dependency it trades in instead:
    correctness now rests entirely on config.SPEC_* being the genuine
    rated range, not a guess — see that docstring.

    No labels used, consistent with the rest of this module.
    """
    mask = (
        (raw_df[config.COL_VIBRATION] <= config.SPEC_VIBRATION_MAX)
        & (raw_df[config.COL_TEMPERATURE] <= config.SPEC_TEMPERATURE_MAX)
        & (raw_df[config.COL_CURRENT] <= config.SPEC_CURRENT_MAX)
    )
    normal_df = raw_df[mask].reset_index(drop=True)

    frac = len(normal_df) / len(raw_df) if len(raw_df) else 0.0
    print(
        f"[select_spec_normal_rows] {len(normal_df)}/{len(raw_df)} rows "
        f"({frac:.1%}) within spec bounds "
        f"(vibration<={config.SPEC_VIBRATION_MAX}, "
        f"temperature<={config.SPEC_TEMPERATURE_MAX}, "
        f"current<={config.SPEC_CURRENT_MAX})."
    )
    if normal_df.empty:
        print(
            "[select_spec_normal_rows] WARNING: zero rows passed the spec "
            "filter. Check config.SPEC_* bounds against actual sensor units "
            "and ranges before proceeding — fit() will fail on an empty set."
        )
    return normal_df


def run_preprocessing(save: bool = True) -> pd.DataFrame:
    df = load_data()
    df = clean_data(df)

    if save:
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
        save_source_signature()
        print(f"[run_preprocessing] Saved cleaned data to {config.PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    run_preprocessing()
