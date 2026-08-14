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
    ready = roll.count() >= min_periods
    rolling_skew = roll.skew().replace([np.inf, -np.inf], np.nan)
    rolling_kurtosis = roll.kurt().replace([np.inf, -np.inf], np.nan)
    rolling_range = roll.max() - roll.min()
    rolling_scale = series.abs().rolling(
        window=window, min_periods=min_periods
    ).max().clip(lower=1.0)
    effectively_constant = (
        rolling_range <= np.finfo(float).eps * rolling_scale * 100
    )
    # pandas' rolling moment implementations are vectorized and return NaN
    # for a constant window.  Preserve NaN while history is incomplete, but
    # represent a ready constant window as zero (the same policy as the scalar
    # safety helpers above) without invoking a Python callback per source row.
    rolling_skew = rolling_skew.where(rolling_skew.notna(), 0.0).where(ready)
    rolling_kurtosis = rolling_kurtosis.where(rolling_kurtosis.notna(), 0.0).where(ready)
    rolling_skew = rolling_skew.where(~effectively_constant, 0.0).where(ready)
    rolling_kurtosis = rolling_kurtosis.where(~effectively_constant, 0.0).where(ready)
    out = pd.DataFrame({
        f"{prefix}_mean": roll.mean(),
        f"{prefix}_std": roll.std(),
        f"{prefix}_max": roll.max(),
        f"{prefix}_min": roll.min(),
        f"{prefix}_rms": np.sqrt(series.square().rolling(window=window, min_periods=min_periods).mean()),
        f"{prefix}_kurtosis": rolling_kurtosis,
        f"{prefix}_skew": rolling_skew,
    })
    # crest factor = peak / rms — classic early-defect vibration indicator
    rms = out[f"{prefix}_rms"]
    out[f"{prefix}_crest_factor"] = (roll.max() / rms.replace(0, np.nan)).where(rms != 0, 0.0)
    return out


def _trend_slope(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Vectorized least-squares slope over a fixed row window.

    The previous rolling.apply(np.polyfit) executed Python once per source row
    and made multi-million-row commissioning scans impractical.  This is the
    algebraically equivalent ordinary-least-squares slope for complete fixed
    windows, using rolling sums implemented by pandas.
    """
    if min_periods != window:
        # The production configuration uses complete windows.  Keep the
        # general fallback correct for offline experiments that deliberately
        # choose a different min_periods value.
        def _slope(values):
            if len(values) < 2 or _effectively_constant(values):
                return 0.0 if len(values) >= 2 else np.nan
            return float(np.polyfit(np.arange(len(values)), values, 1)[0])
        return series.rolling(window=window, min_periods=min_periods).apply(_slope, raw=True)

    positions = pd.Series(np.arange(len(series), dtype=float), index=series.index)
    rolling = series.rolling(window=window, min_periods=min_periods)
    sum_y = rolling.sum()
    sum_xy_global = (series * positions).rolling(
        window=window, min_periods=min_periods
    ).sum()
    window_start = positions - (window - 1)
    sum_xy_local = sum_xy_global - window_start * sum_y
    sum_x = window * (window - 1) / 2.0
    sum_x2 = window * (window - 1) * (2 * window - 1) / 6.0
    denominator = window * sum_x2 - sum_x * sum_x
    slope = (window * sum_xy_local - sum_x * sum_y) / denominator
    ready = rolling.count() >= min_periods
    rolling_range = rolling.max() - rolling.min()
    rolling_scale = series.abs().rolling(
        window=window, min_periods=min_periods
    ).max().clip(lower=1.0)
    effectively_constant = (
        rolling_range <= np.finfo(float).eps * rolling_scale * 100
    )
    return slope.where(~effectively_constant, 0.0).where(ready)


def create_features_chunk(
    clean_rows: pd.DataFrame,
    carry_rows: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Engineer one clean chronological chunk with exact rolling continuity.

    Only the final WINDOW_SIZE-1 clean source rows are carried between chunks.
    Returned features belong exclusively to ``clean_rows``; carry rows are
    context and are never emitted twice.
    """
    clean_rows = clean_rows.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
    carry_rows = carry_rows if carry_rows is not None else clean_rows.iloc[0:0]
    combined = pd.concat([carry_rows, clean_rows], ignore_index=True)
    featured = create_features(combined, verbose=False)
    if len(clean_rows):
        current_timestamps = set(clean_rows[config.COL_TIMESTAMP])
        featured = featured[
            featured[config.COL_TIMESTAMP].isin(current_timestamps)
        ].reset_index(drop=True)
    else:
        featured = featured.iloc[0:0].copy()
    carry_count = max(0, config.WINDOW_SIZE - 1)
    next_carry = combined.tail(carry_count).copy().reset_index(drop=True)
    return featured, next_carry


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
        correlation = rolling_a.corr(df[col_b]).replace([np.inf, -np.inf], np.nan)
        ready = (rolling_a.count() >= mp) & (rolling_b.count() >= mp)
        std_a = rolling_a.std()
        std_b = rolling_b.std()
        scale_a = df[col_a].abs().rolling(window=w, min_periods=mp).max().clip(lower=1.0)
        scale_b = df[col_b].abs().rolling(window=w, min_periods=mp).max().clip(lower=1.0)
        tolerance_a = np.finfo(float).eps * scale_a * 100
        tolerance_b = np.finfo(float).eps * scale_b * 100
        stable = ready & (std_a > tolerance_a) & (std_b > tolerance_b)

        # Pearson correlation is undefined for constant windows and can become
        # +/-Inf through catastrophic cancellation for almost-flat industrial
        # channels.  Zero means "no measurable linear relationship" here; the
        # separate std/range features still retain the important flatness
        # signal.  Clip tiny numerical overshoots back to the legal range.
        correlation = correlation.where(stable, 0.0).fillna(0.0).clip(-1.0, 1.0)
        corr_frame[f"{col_a}_{col_b}_corr"] = correlation
    feat_frames.append(pd.DataFrame(corr_frame))

    result = pd.concat(feat_frames, axis=1)

    n_before = len(result)
    result = result.dropna().reset_index(drop=True)
    n_dropped = n_before - len(result)
    if n_dropped and verbose:
        print(f"[create_features] Dropped {n_dropped} rows with incomplete rolling windows "
              f"(insufficient history at trajectory start).")

    feature_columns = [column for column in result.columns if column != config.COL_TIMESTAMP]
    if feature_columns:
        values = result[feature_columns].to_numpy(dtype=float)
        finite = np.isfinite(values)
        if not finite.all():
            bad_columns = [
                column for index, column in enumerate(feature_columns)
                if not finite[:, index].all()
            ]
            raise ValueError(
                "create_features() produced non-finite values in "
                f"{bad_columns}. Refusing to pass invalid features downstream."
            )

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
