"""
Loading and cleaning for the spindle condition-monitoring pipeline.

CRITICAL: load_data() reads ONLY [timestamp] + RAW_SENSOR_COLS via
pandas' `usecols`. This is deliberate and structural, not just a
convention to remember — health_status exists in the raw CSV but must
never enter this codebase (see config.py's module docstring for why).
Using `usecols` means even a future column-selection bug elsewhere can't
accidentally pull it in; it's dropped before anything else runs.
"""

import pandas as pd

import config


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


def run_preprocessing(save: bool = True) -> pd.DataFrame:
    df = load_data()
    df = clean_data(df)

    if save:
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
        print(f"[run_preprocessing] Saved cleaned data to {config.PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    run_preprocessing()
