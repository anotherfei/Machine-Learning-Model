"""
Trend forecasting — a separate, swappable stage from the Kalman filter
(see kalman.py docstring for why). Fits a simple linear regression over
the last TREND_LOOKBACK_MINUTES of Kalman-smoothed health values and
extrapolates forward to when health crosses FAILURE_HEALTH_THRESHOLD.

Linear degradation is the simplest reasonable default (this matches how
the article's walkthrough frames it) and is what config.py documents as
the assumption in use. If a given asset's real degradation is closer to
exponential/accelerating wear, swap fit_trend()'s model — the rest of the
pipeline (failure_probability.py, maintenance.py) depends on the fitted
slope/intercept plus a condition-diffusion estimate derived here.
"""

import numpy as np

import config
import runtime_config


def fit_trend(minutes_since_start: np.ndarray, health_values: np.ndarray):
    """
    Least-squares linear fit: health ~ a * minutes + b.
    Returns (slope_per_minute, intercept, residual_std) or None if there
    aren't enough points yet (see config.TREND_MIN_POINTS).
    """
    minimum_points = runtime_config.get("TREND_MIN_POINTS", config.TREND_MIN_POINTS)
    if len(minutes_since_start) < minimum_points:
        return None

    slope, intercept = np.polyfit(minutes_since_start, health_values, 1)
    predicted = slope * minutes_since_start + intercept
    residual_std = float(np.std(health_values - predicted)) or 1e-6

    return slope, intercept, residual_std


def slope_is_significant(minutes_since_start: np.ndarray, slope: float, residual_std: float,
                          z_threshold: float = None) -> bool:
    """
    Standard OLS test of whether `slope` is distinguishable from noise,
    given the fit it came from. This is the actual fix for "remaining_days
    is sensitive to slope noise" (see config.TREND_SLOPE_Z_THRESHOLD): it
    doesn't matter whether the fit is early or late in the trajectory —
    what matters is whether the estimated slope is large relative to its
    own uncertainty. A shallow, noisy fit and a genuinely-declining trend
    can produce the same point-estimate slope; this is what tells them
    apart.

    SE(slope) = s / sqrt(Sxx), the standard formula for simple linear
    regression, where s is the residual standard error (residual_std
    already computed with ddof=0 in fit_trend, corrected here to ddof=2
    — negligible at TREND_MIN_POINTS=60+, done properly anyway) and Sxx
    is the sum of squared deviations of `minutes_since_start` from its
    mean. |slope / SE(slope)| >= z_threshold is the significance test.

    Validated against data/raw/spindle.csv: with z_threshold=2.0 (~95%,
    the standard two-tailed convention — not tuned against this dataset
    specifically), this removes 874 of 1018 "CRITICAL while health_state
    is still in the 90s" rows found after the Kalman warm-up fix — all of
    them noise-driven (small/insignificant slopes that only crossed the
    remaining_days<=7 cutoff because remaining_days() floors at 1 day).
    The remaining 144 are unaffected — they match the original,
    unmodified pipeline's count exactly, meaning they're the same
    genuinely-significant declining trends that were always there, not
    something this check happens to also suppress.

    Returns False (untrusted) for degenerate inputs (n<=2, zero variance
    in x, zero standard error) rather than raising — callers already
    treat "not trusted" as the safe default via maintenance.recommend()'s
    trend_trusted parameter.
    """
    z_threshold = (
        runtime_config.get("TREND_SLOPE_Z_THRESHOLD", config.TREND_SLOPE_Z_THRESHOLD)
        if z_threshold is None else z_threshold
    )
    n = len(minutes_since_start)
    if n <= 2:
        return False

    x = np.asarray(minutes_since_start, dtype=float)
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 0:
        return False

    residual_se = residual_std * np.sqrt(n / (n - 2))  # ddof=0 -> ddof=2 correction
    slope_se = residual_se / np.sqrt(sxx)
    if slope_se <= 0:
        return False

    return abs(slope / slope_se) >= z_threshold


def estimate_diffusion(minutes_since_start: np.ndarray, health_values: np.ndarray,
                       slope: float, intercept: float) -> float:
    """Estimate condition innovation scale in health points / sqrt(minute).

    Regression residual level is not a per-step random-walk noise parameter.
    Estimate diffusion from consecutive detrended residual innovations and
    their actual timestamp gaps instead. A robust MAD estimate limits the
    influence of isolated spikes; RMS is a fallback for quantized series.
    """
    times = np.asarray(minutes_since_start, dtype=float)
    values = np.asarray(health_values, dtype=float)
    if len(times) < 3 or len(times) != len(values):
        return 1e-6
    residuals = values - (float(slope) * times + float(intercept))
    intervals = np.diff(times)
    innovations = np.diff(residuals)
    valid = np.isfinite(intervals) & np.isfinite(innovations) & (intervals > 0)
    if not valid.any():
        return 1e-6
    standardized = innovations[valid] / np.sqrt(intervals[valid])
    center = float(np.median(standardized))
    robust = float(np.median(np.abs(standardized - center)) * 1.4826)
    rms = float(np.sqrt(np.mean(np.square(standardized))))
    estimate = robust if np.isfinite(robust) and robust > 1e-12 else rms
    return max(float(estimate) if np.isfinite(estimate) else 0.0, 1e-6)


def remaining_days(current_minute: float, current_health: float, slope_per_minute: float) -> int:
    """
    At the current (linear) rate of change, how many days until health
    crosses FAILURE_HEALTH_THRESHOLD? Floored/capped the same way as the
    earlier sensor-trend version: a flat or improving trend reports the
    cap, not infinity.
    """
    failure_threshold = runtime_config.get(
        "FAILURE_HEALTH_THRESHOLD", config.FAILURE_HEALTH_THRESHOLD
    )
    if slope_per_minute >= -1e-6 or current_health <= failure_threshold:
        return 0 if current_health <= failure_threshold else config.REMAINING_DAYS_CAP

    minutes_to_threshold = (failure_threshold - current_health) / slope_per_minute
    days = minutes_to_threshold / (24 * 60)
    return int(np.clip(days, 1, config.REMAINING_DAYS_CAP))
