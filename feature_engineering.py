"""
Rolling/statistical feature engineering for the Gearbox RUL pipeline.

Every feature is computed per-unit (grouped by unit_id) so that a rolling
window never spans two different gearboxes. Windows with insufficient
history produce NaN (via min_periods) — these are dropped explicitly and
loudly, never silently imputed, because a manufactured value here would
corrupt the degradation signal right at the point it matters most.
"""

import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

import config


def _rolling_stats(series: pd.Series, window: int, min_periods: int, prefix: str) -> pd.DataFrame:
    roll = series.rolling(window=window, min_periods=min_periods)
    out = pd.DataFrame({
        f"{prefix}_mean": roll.mean(),
        f"{prefix}_std": roll.std(),
        f"{prefix}_max": roll.max(),
        f"{prefix}_min": roll.min(),
        f"{prefix}_rms": roll.apply(lambda x: np.sqrt(np.mean(np.square(x))), raw=True),
        f"{prefix}_kurtosis": roll.apply(lambda x: kurtosis(x, bias=False) if len(x) > 3 else np.nan, raw=True),
        f"{prefix}_skew": roll.apply(lambda x: skew(x, bias=False) if len(x) > 2 else np.nan, raw=True),
    })
    # crest factor = peak / rms — a classic vibration health indicator
    out[f"{prefix}_crest_factor"] = roll.max() / out[f"{prefix}_rms"].replace(0, np.nan)
    return out


def _trend_slope(series: pd.Series, window: int, min_periods: int) -> pd.Series:
    """Rolling linear-fit slope — captures rate of change, not just level."""
    def _slope(x):
        if len(x) < 2 or np.all(x == x[0]):
            return np.nan
        idx = np.arange(len(x))
        return np.polyfit(idx, x, 1)[0]

    return series.rolling(window=window, min_periods=min_periods).apply(_slope, raw=True)


def _create_features_for_unit(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values(config.COL_TIMESTAMP).copy()
    w, mp = config.WINDOW_SIZE, config.MIN_PERIODS

    feat_frames = [g[[config.COL_UNIT_ID, config.COL_TIMESTAMP]]]

    # Vibration — full stat suite, this is the most diagnostic signal for gearboxes
    feat_frames.append(_rolling_stats(g[config.COL_VIBRATION], w, mp, "vibration"))

    # Current — level + trend, current draw rises with mechanical resistance
    feat_frames.append(_rolling_stats(g[config.COL_CURRENT], w, mp, "current"))
    feat_frames.append(pd.DataFrame({
        "current_trend_slope": _trend_slope(g[config.COL_CURRENT], w, mp)
    }))

    # Temperature — level + rate of rise, lagging indicator, confirms severity
    feat_frames.append(_rolling_stats(g[config.COL_TEMPERATURE], w, mp, "temperature"))
    feat_frames.append(pd.DataFrame({
        "temperature_trend_slope": _trend_slope(g[config.COL_TEMPERATURE], w, mp)
    }))

    # Cross-signal: rolling correlation between vibration and current —
    # divergence from baseline correlation often precedes failure
    corr = g[config.COL_VIBRATION].rolling(window=w, min_periods=mp).corr(g[config.COL_CURRENT])
    feat_frames.append(pd.DataFrame({"vibration_current_corr": corr}))

    if config.COL_RUL in g.columns:
        feat_frames.append(g[[config.COL_RUL]])

    result = pd.concat(feat_frames, axis=1)
    return result


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    # Iterate explicitly rather than groupby(...).apply(...) — apply can
    # silently drop the grouping column in some pandas versions when the
    # returned frame still contains it (see preprocessing.py for the same fix).
    feature_frames = [_create_features_for_unit(g) for _, g in df.groupby(config.COL_UNIT_ID)]
    df = pd.concat(feature_frames, ignore_index=True)

    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        print(f"[create_features] Dropped {n_dropped} rows with incomplete rolling windows "
              f"(insufficient history at trajectory start).")

    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Every column that is NOT an identifier or the target."""
    exclude = {config.COL_UNIT_ID, config.COL_TIMESTAMP, config.COL_RUL}
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
