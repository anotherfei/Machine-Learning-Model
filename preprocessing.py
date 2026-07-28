"""
Cleaning and splitting for the Gearbox RUL pipeline.

CRITICAL: splitting is done by unit_id, never by row. Random row splits leak
adjacent-in-time windows from the same degradation trajectory into both train
and test, which inflates every downstream metric and invalidates model
comparisons. Do not "simplify" this file by switching to train_test_split
on rows.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

import config


def load_data(path: str = config.RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)

    required = [config.COL_UNIT_ID, config.COL_TIMESTAMP] + config.RAW_SENSOR_COLS
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"Raw CSV is missing required columns: {missing}. "
            f"Check config.py column names against your actual CSV headers."
        )
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Parse timestamp, sort within each unit — order matters for rolling features
    df[config.COL_TIMESTAMP] = pd.to_datetime(df[config.COL_TIMESTAMP], errors="coerce")
    df = df.sort_values([config.COL_UNIT_ID, config.COL_TIMESTAMP])

    # Drop exact duplicate rows
    before = len(df)
    df = df.drop_duplicates(subset=[config.COL_UNIT_ID, config.COL_TIMESTAMP])
    dropped = before - len(df)
    if dropped:
        print(f"[clean_data] Dropped {dropped} duplicate rows.")

    # Drop rows with missing sensor values rather than imputing silently —
    # imputing rolling sensor data (e.g. ffill) can manufacture a false
    # "flat" period that looks like healthy operation. If you need imputation,
    # do it deliberately here and document why.
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


def label_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensures a RUL column exists. If config.PIECEWISE_LABELING is True, derives
    RUL from each unit's max timestamp/cycle instead of trusting a raw column
    (use this if your CSV gives failure timestamps but not RUL directly).
    """
    df = df.copy()

    if config.COL_RUL in df.columns and not config.PIECEWISE_LABELING:
        return df

    if config.COL_RUL not in df.columns and not config.PIECEWISE_LABELING:
        raise ValueError(
            f"No '{config.COL_RUL}' column found and PIECEWISE_LABELING is False. "
            f"Either add a RUL column to the CSV or set PIECEWISE_LABELING=True "
            f"in config.py so RUL is derived from cycle position."
        )

    # Piecewise-linear RUL: cycles_to_failure, capped at RUL_CAP
    def _label_unit(g):
        g = g.sort_values(config.COL_TIMESTAMP)
        n = len(g)
        cycles_to_failure = np.arange(n - 1, -1, -1)
        g[config.COL_RUL] = np.minimum(cycles_to_failure, config.RUL_CAP)
        return g

    # NOTE: groupby(...).apply(...) can silently drop the grouping column in
    # some pandas versions when the applied function returns a frame that
    # still contains it. Iterate explicitly and concat instead, so unit_id
    # is never at risk of disappearing from the output.
    labeled = [_label_unit(g) for _, g in df.groupby(config.COL_UNIT_ID)]
    df = pd.concat(labeled, ignore_index=True)
    return df


def split_data(df: pd.DataFrame, test_size: float = config.TEST_UNIT_FRACTION):
    """
    Group-aware holdout split — entire units go to either train or test.
    For model comparison, prefer GroupKFold in compare_models.py over this;
    this function is for a quick single train/test split (e.g. train.py).
    """
    groups = df[config.COL_UNIT_ID]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=config.RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(df, groups=groups))

    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)

    train_units = set(train_df[config.COL_UNIT_ID].unique())
    test_units = set(test_df[config.COL_UNIT_ID].unique())
    overlap = train_units & test_units
    assert not overlap, f"Unit leakage between train/test: {overlap}"

    print(f"[split_data] Train units: {len(train_units)}, Test units: {len(test_units)}")
    return train_df, test_df


def run_preprocessing(save: bool = True) -> pd.DataFrame:
    df = load_data()
    df = clean_data(df)
    df = label_rul(df)

    if save:
        df.to_csv(config.PROCESSED_DATA_PATH, index=False)
        print(f"[run_preprocessing] Saved cleaned data to {config.PROCESSED_DATA_PATH}")

    return df


if __name__ == "__main__":
    run_preprocessing()
