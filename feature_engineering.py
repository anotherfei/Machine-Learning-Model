"""
Rolling/statistical feature engineering for the spindle condition-
monitoring pipeline. Purely a function of the sensor columns — no status
or label column is ever read or produced here (see config.py docstring).

Windows with insufficient history produce NaN (via min_periods) and are
dropped explicitly, never imputed — a manufactured value here would
corrupt the exact early-precursor signal the Isolation Forest is meant to
pick up on.
"""

import itertools

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

import config


def _effectively_constant(values) -> bool:
    """Whether a window has too little variation for stable moments.

    Industrial channels are often quantized and can remain exactly or almost
    flat for many samples. Passing those values to scipy.stats skew/kurtosis
    causes catastrophic-cancellation warnings and produces unstable features.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return True
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = np.finfo(float).eps * scale * 100
    return float(np.ptp(values)) <= tolerance


def _safe_kurtosis(values) -> float:
    if len(values) <= 3 or _effectively_constant(values):
        return 0.0
    value = float(kurtosis(values, bias=False))
    return value if np.isfinite(value) else 0.0


def _safe_skew(values) -> float:
    if len(values) <= 2 or _effectively_constant(values):
        return 0.0
    value = float(skew(values, bias=False))
    return value if np.isfinite(value) else 0.0


def _rolling_stats(series: pd.Series, window: int, min_periods: int, prefix: str) -> pd.DataFrame:
    roll = series.rolling(window=window, min_periods=min_periods)
    out = pd.DataFrame({
        f"{prefix}_mean": roll.mean(),
        f"{prefix}_std": roll.std(),
        f"{prefix}_max": roll.max(),
        f"{prefix}_min": roll.min(),
        f"{prefix}_rms": roll.apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True),
        f"{prefix}_kurtosis": roll.apply(_safe_kurtosis, raw=True),
        f"{prefix}_skew": roll.apply(_safe_skew, raw=True),
    })
    # crest factor = peak / rms — classic early-defect vibration indicator
    rms = out[f"{prefix}_rms"]
    out[f"{prefix}_crest_factor"] = (roll.max() / rms.replace(0, np.nan)).where(rms != 0, 0.0)
    return out


def _trend_slope(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    def _slope(x):
        if len(x) < 2:
            return np.nan
        if _effectively_constant(x):
            return 0.0
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]

    return series.rolling(window=window, min_periods=min_periods).apply(_slope, raw=True)


def create_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    df = df.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
    w, mp = config.WINDOW_SIZE, config.MIN_PERIODS
    sensor_cols = config.RAW_SENSOR_COLS

    feat_frames = [df[[config.COL_TIMESTAMP]]]

    # Rolling stats + trend slope, generic over whatever raw sensor
    # columns config.py currently defines (previously hardcoded to
    # vibration/current/temperature specifically — genericized so a
    # sensor swap only requires updating config.RAW_SENSOR_COLS, not
    # this function).
    for col in sensor_cols:
        feat_frames.append(_rolling_stats(df[col], w, mp, col))
        feat_frames.append(pd.DataFrame({
            f"{col}_trend_slope": _trend_slope(df[col], w, mp)
        }))

    # Cross-sensor rolling correlations, every pair — was hardcoded to
    # the 3 pairs available under the old 3-column schema (vibration x
    # current, vibration x temperature, current x temperature); with 5
    # raw columns now that's all 10 pairs instead of 3.
    corr_frame = {}
    for col_a, col_b in itertools.combinations(sensor_cols, 2):
        rolling_a = df[col_a].rolling(window=w, min_periods=mp)
        rolling_b = df[col_b].rolling(window=w, min_periods=mp)
        correlation = rolling_a.corr(df[col_b])
        ready = (rolling_a.count() >= mp) & (rolling_b.count() >= mp)
        # Correlation is undefined when either complete window is constant.
        # Treat that as no measured linear relationship rather than dropping
        # an otherwise valid production tick from the feature table.
        correlation = correlation.mask(ready & correlation.isna(), 0.0)
        corr_frame[f"{col_a}_{col_b}_corr"] = correlation
    feat_frames.append(pd.DataFrame(corr_frame))

    result = pd.concat(feat_frames, axis=1)

    n_before = len(result)
    result = result.dropna().reset_index(drop=True)
    n_dropped = n_before - len(result)
    if n_dropped and verbose:
        print(f"[create_features] Dropped {n_dropped} rows with incomplete rolling windows "
              f"(insufficient history at trajectory start).")

    return result


def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = {config.COL_TIMESTAMP}
    return [c for c in df.columns if c not in exclude]


def run_feature_engineering(save: bool = True) -> pd.DataFrame:
    df = pd.read_csv(config.PROCESSED_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])
    df = create_features(df)

    if save:
        df.to_csv(config.FEATURES_DATA_PATH, index=False)
        print(f"[run_feature_engineering] Saved features to {config.FEATURES_DATA_PATH}")
        print(f"[run_feature_engineering] {len(get_feature_columns(df))} feature columns.")

    return df


if __name__ == "__main__":
    run_feature_engineering()
