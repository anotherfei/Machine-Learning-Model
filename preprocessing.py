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
import artifact_utils


def raw_data_signature(path: str = None) -> str:
    """
    Fingerprint of everything that determines what data/processed/*.csv
    should contain: the raw file's own content, PLUS every config value
    that changes how it gets processed. A cache built before any of
    these changed is stale even if the raw file itself is untouched.

    Covers:
      - the raw CSV's content (catches RAW_DATA_PATH pointing elsewhere,
        or the same file edited in place)
      - COL_TIMESTAMP / RAW_SENSOR_COLS — change any of these and
        load_data()/clean_data()'s behavior changes, even though
        load_data() itself wasn't touched
      - artifact_utils.config_hash() — FEATURE_CONFIG (WINDOW_SIZE,
        MIN_PERIODS, ...) plus feature_engineering.py's own source; reused
        rather than reimplemented so this and the model's own drift check
        can never disagree about what counts as a config change

    Content-hashed, not path+size+mtime (an earlier version of this
    function used that) — mtime is not reliable enough to gate a silent
    skip-vs-rebuild decision on: cloud-synced folders (OneDrive, Dropbox —
    common under Windows' Documents/) can leave mtime stale after a sync
    even when content changed, and some filesystems have coarser mtime
    granularity than a fast edit-save-rerun loop needs.
    """
    path = path or config.RAW_DATA_PATH
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            hasher.update(chunk)

    relevant_config = {
        "COL_TIMESTAMP": config.COL_TIMESTAMP,
        "RAW_SENSOR_COLS": config.RAW_SENSOR_COLS,
    }
    hasher.update(json.dumps(relevant_config, sort_keys=True).encode())
    hasher.update(artifact_utils.config_hash().encode())
    return hasher.hexdigest()


def save_source_signature(path: str = None):
    sig = raw_data_signature(path)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)
    with open(os.path.join(config.PROCESSED_DATA_DIR, "source_signature.json"), "w") as f:
        json.dump({"source_path": os.path.abspath(path or config.RAW_DATA_PATH), "signature": sig}, f, indent=2)


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

    Always prints its decision and both signatures/paths compared, so a
    skip-vs-rebuild call is never a silent black box — if this looks
    wrong on your machine, the printed paths/hashes are the first thing
    to check, not this function's logic.
    """
    current_path = os.path.abspath(path or config.RAW_DATA_PATH)
    if not os.path.exists(config.FEATURES_DATA_PATH):
        print(f"[cached_features_are_stale] {config.FEATURES_DATA_PATH} does not exist -> STALE (will rebuild).")
        return True
    sig_path = os.path.join(config.PROCESSED_DATA_DIR, "source_signature.json")
    if not os.path.exists(sig_path):
        print(f"[cached_features_are_stale] No {sig_path} found -> STALE (will rebuild).")
        return True

    with open(sig_path) as f:
        saved = json.load(f)
    current_sig = raw_data_signature(current_path)
    is_stale = saved.get("signature") != current_sig

    print(f"[cached_features_are_stale] Cached source: {saved.get('source_path')} "
          f"(hash {str(saved.get('signature'))[:12]}...)")
    print(f"[cached_features_are_stale] Current RAW_DATA_PATH: {current_path} (hash {current_sig[:12]}...)")
    print(f"[cached_features_are_stale] -> {'STALE (will rebuild)' if is_stale else 'up to date (using cache)'}.")
    return is_stale


def load_data(path: str = None) -> pd.DataFrame:
    """
    path defaults to config.RAW_DATA_PATH, resolved INSIDE the function
    body (not as the parameter default) so it always reflects the
    CURRENT value of config.RAW_DATA_PATH at call time. A parameter
    default of `config.RAW_DATA_PATH` (an earlier version of this
    function had that) is evaluated exactly once, when this module is
    first imported — if config.RAW_DATA_PATH changes afterward in a
    process that keeps this import alive (a persistent notebook kernel
    or IDE terminal, for instance), load_data() called with no explicit
    path would silently keep using whatever RAW_DATA_PATH was at import
    time, not the current one. That's the kind of bug that looks like
    "inconsistent, no clear pattern" from the outside.
    """
    path = path or config.RAW_DATA_PATH
    usecols = [config.COL_TIMESTAMP] + config.RAW_SENSOR_COLS
    df = pd.read_csv(path, usecols=usecols)

    missing = [c for c in usecols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw CSV is missing required columns: {missing}. "
            f"Check config.py column names against your actual CSV headers."
        )
    return df


def clean_data(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.copy()

    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP], errors="coerce")
    df = df.sort_values(config.COL_TIMESTAMP)

    before = len(df)
    df = df.dropna(subset=[config.COL_TIMESTAMP])
    dropped = before - len(df)
    if dropped and verbose:
        print(f"[clean_data] Dropped {dropped} rows with invalid timestamps.")

    before = len(df)
    df = df.drop_duplicates(subset=[config.COL_TIMESTAMP])
    dropped = before - len(df)
    if dropped and verbose:
        print(f"[clean_data] Dropped {dropped} duplicate timestamp rows.")

    before = len(df)
    df = df.dropna(subset=config.RAW_SENSOR_COLS)
    dropped = before - len(df)
    if dropped and verbose:
        print(f"[clean_data] Dropped {dropped} rows with missing sensor values.")

    # Remove physically impossible readings, per the VVB001 datasheet's
    # own measuring ranges (not a guess — see product-spec conversation):
    #   a-RMS / a-Peak: 0-490.3 m/s^2 (0-50 g)
    #   v-RMS:          0-45 mm/s
    #   crest factor:   1-50 (also mathematically >=1 always: peak>=rms
    #                   for any real signal, by definition of RMS)
    #   temperature:    -30 to 80 C (sensor's rated measuring range)
    df = df[
        (df[config.COL_A_RMS] >= 0) & (df[config.COL_A_RMS] <= 490.3)
        & (df[config.COL_A_PEAK] >= 0) & (df[config.COL_A_PEAK] <= 490.3)
        & (df[config.COL_V_RMS] >= 0) & (df[config.COL_V_RMS] <= 45)
        & (df[config.COL_CREST_FACTOR] >= 1) & (df[config.COL_CREST_FACTOR] <= 50)
        & (df[config.COL_TEMPERATURE] >= -30) & (df[config.COL_TEMPERATURE] <= 80)
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
    active_bounds = {col: bound for col, bound in config.SPEC_MAX.items() if bound is not None}
    if not active_bounds:
        raise ValueError(
            "select_spec_normal_rows(): every config.SPEC_MAX bound is None — "
            "nothing to filter on. Set at least one SPEC_*_MAX in config.py, "
            "or use split_reference_window() instead."
        )

    mask = pd.Series(True, index=raw_df.index)
    for col, bound in active_bounds.items():
        mask &= raw_df[col] <= bound
    normal_df = raw_df[mask].reset_index(drop=True)

    frac = len(normal_df) / len(raw_df) if len(raw_df) else 0.0
    bounds_str = ", ".join(f"{col}<={bound}" for col, bound in active_bounds.items())
    skipped = [col for col, bound in config.SPEC_MAX.items() if bound is None]
    print(
        f"[select_spec_normal_rows] {len(normal_df)}/{len(raw_df)} rows "
        f"({frac:.1%}) within spec bounds ({bounds_str})."
    )
    if skipped:
        print(f"[select_spec_normal_rows] No bound set (skipped) for: {', '.join(skipped)}.")
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
