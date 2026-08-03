"""
Trend forecasting — a separate, swappable stage from the Kalman filter
(see kalman.py docstring for why). Fits a simple linear regression over
the last TREND_LOOKBACK_MINUTES of Kalman-smoothed health values and
extrapolates forward to when health crosses FAILURE_HEALTH_THRESHOLD.

Linear degradation is the simplest reasonable default (this matches how
the article's walkthrough frames it) and is what config.py documents as
the assumption in use. If a given asset's real degradation is closer to
exponential/accelerating wear, swap fit_trend()'s model — the rest of the
pipeline (failure_probability.py, maintenance.py) only depends on
(slope, intercept, residual_std), not on linear regression specifically.
"""

import numpy as np

import config


def fit_trend(minutes_since_start: np.ndarray, health_values: np.ndarray):
    """
    Least-squares linear fit: health ~ a * minutes + b.
    Returns (slope_per_minute, intercept, residual_std) or None if there
    aren't enough points yet (see config.TREND_MIN_POINTS).
    """
    if len(minutes_since_start) < config.TREND_MIN_POINTS:
        return None

    slope, intercept = np.polyfit(minutes_since_start, health_values, 1)
    predicted = slope * minutes_since_start + intercept
    residual_std = float(np.std(health_values - predicted)) or 1e-6

    return slope, intercept, residual_std


def remaining_days(current_minute: float, current_health: float, slope_per_minute: float) -> int:
    """
    At the current (linear) rate of change, how many days until health
    crosses FAILURE_HEALTH_THRESHOLD? Floored/capped the same way as the
    earlier sensor-trend version: a flat or improving trend reports the
    cap, not infinity.
    """
    if slope_per_minute >= -1e-6 or current_health <= config.FAILURE_HEALTH_THRESHOLD:
        return 0 if current_health <= config.FAILURE_HEALTH_THRESHOLD else config.REMAINING_DAYS_CAP

    minutes_to_threshold = (config.FAILURE_HEALTH_THRESHOLD - current_health) / slope_per_minute
    days = minutes_to_threshold / (24 * 60)
    return int(np.clip(days, 1, config.REMAINING_DAYS_CAP))
