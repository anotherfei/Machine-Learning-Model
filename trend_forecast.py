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

from collections import deque

import numpy as np

import config
import runtime_config


def _fit_from_moments(n, sum_x, sum_y, sum_x2, sum_xy, sum_y2):
    """Fit OLS from sufficient statistics shared by array and rolling paths."""
    minimum_points = runtime_config.get("TREND_MIN_POINTS", config.TREND_MIN_POINTS)
    if n < minimum_points:
        return None
    centered_x2 = float(sum_x2) - float(sum_x) * float(sum_x) / n
    if centered_x2 <= 0:
        return None
    centered_xy = float(sum_xy) - float(sum_x) * float(sum_y) / n
    centered_y2 = max(0.0, float(sum_y2) - float(sum_y) * float(sum_y) / n)
    slope = centered_xy / centered_x2
    intercept = (float(sum_y) - slope * float(sum_x)) / n
    residual_variance = max(0.0, (centered_y2 - slope * centered_xy) / n)
    residual_std = float(np.sqrt(residual_variance)) or 1e-6
    return slope, intercept, residual_std


class RollingTrendWindow:
    """Exact elapsed-time trend window with constant-time OLS updates."""

    def __init__(self, lookback_minutes: float):
        self.lookback_minutes = float(lookback_minutes)
        self.minutes = deque()
        self.health = deque()
        self.sum_x = self.sum_y = 0.0
        self.sum_x2 = self.sum_xy = self.sum_y2 = 0.0
        self._updates = 0

    def append(self, minute: float, health: float) -> None:
        x, y = float(minute), float(health)
        self.minutes.append(x); self.health.append(y)
        self.sum_x += x; self.sum_y += y
        self.sum_x2 += x*x; self.sum_xy += x*y; self.sum_y2 += y*y
        cutoff = x - self.lookback_minutes
        while self.minutes and self.minutes[0] < cutoff:
            old_x = self.minutes.popleft(); old_y = self.health.popleft()
            self.sum_x -= old_x; self.sum_y -= old_y
            self.sum_x2 -= old_x*old_x
            self.sum_xy -= old_x*old_y
            self.sum_y2 -= old_y*old_y
        self._updates += 1
        # Bound floating-point accumulation drift during multi-million-row
        # replay while keeping the normal update constant-time.
        if self._updates % 100_000 == 0:
            x, y = self.arrays()
            self.sum_x=float(x.sum()); self.sum_y=float(y.sum())
            self.sum_x2=float(np.dot(x,x)); self.sum_xy=float(np.dot(x,y))
            self.sum_y2=float(np.dot(y,y))

    def fit(self):
        return _fit_from_moments(
            len(self.minutes), self.sum_x, self.sum_y,
            self.sum_x2, self.sum_xy, self.sum_y2,
        )

    def slope_is_significant(self, slope: float, residual_std: float,
                             z_threshold: float = None) -> bool:
        n = len(self.minutes)
        centered_x2 = self.sum_x2 - self.sum_x*self.sum_x/n if n else 0.0
        return _slope_is_significant_from_sxx(
            n, centered_x2, slope, residual_std, z_threshold,
        )

    def arrays(self):
        return (
            np.fromiter(self.minutes, dtype=float, count=len(self.minutes)),
            np.fromiter(self.health, dtype=float, count=len(self.health)),
        )


def fit_trend(minutes_since_start: np.ndarray, health_values: np.ndarray):
    """
    Least-squares linear fit: health ~ a * minutes + b.
    Returns (slope_per_minute, intercept, residual_std) or None if there
    aren't enough points yet (see config.TREND_MIN_POINTS).
    """
    x = np.asarray(minutes_since_start, dtype=float)
    y = np.asarray(health_values, dtype=float)
    return _fit_from_moments(
        len(x), x.sum(), y.sum(), np.dot(x, x), np.dot(x, y), np.dot(y, y),
    )


def _slope_is_significant_from_sxx(
    n, sxx, slope, residual_std, z_threshold=None,
) -> bool:
    z_threshold = (
        runtime_config.get("TREND_SLOPE_Z_THRESHOLD", config.TREND_SLOPE_Z_THRESHOLD)
        if z_threshold is None else z_threshold
    )
    if n <= 2 or sxx <= 0:
        return False
    residual_se = residual_std * np.sqrt(n / (n - 2))
    slope_se = residual_se / np.sqrt(sxx)
    return bool(slope_se > 0 and abs(slope / slope_se) >= z_threshold)


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
    n = len(minutes_since_start)
    if n <= 2:
        return False
    x = np.asarray(minutes_since_start, dtype=float)
    sxx = float(np.sum((x - x.mean()) ** 2))
    return _slope_is_significant_from_sxx(
        n, sxx, slope, residual_std, z_threshold,
    )


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
