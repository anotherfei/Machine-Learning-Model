"""
Failure probability — turns the trend forecast's point estimate into a
probability distribution over when failure happens, instead of a single
number pretending to be certain.

Model: health evolves as a random walk with drift (standard assumption in
remaining-life / Wiener-process degradation literature). Forecast
uncertainty at horizon h therefore grows with sqrt(h), using the trend
fit's residual std as the per-minute noise scale. This is a real
statistical model, not an arbitrary confidence band — but it inherits
every assumption of the linear trend fit underneath it (see
trend_forecast.py), so treat it as a reasonable approximation, not a
calibrated guarantee, especially far from FAILURE_HEALTH_THRESHOLD where
real degradation curves are less likely to stay linear.
"""

import numpy as np
from scipy.stats import norm

import config


def failure_probability_at(current_health: float, slope_per_minute: float,
                            residual_std: float, horizon_days: float) -> float:
    """P(health has crossed FAILURE_HEALTH_THRESHOLD by horizon_days from now)."""
    horizon_minutes = horizon_days * 24 * 60
    predicted_health = current_health + slope_per_minute * horizon_minutes

    # Uncertainty grows with sqrt(time) under a random-walk-with-drift model
    std_at_horizon = residual_std * np.sqrt(max(horizon_minutes, 1.0))

    z = (config.FAILURE_HEALTH_THRESHOLD - predicted_health) / std_at_horizon
    return float(norm.cdf(z))


def failure_probability_table(current_health: float, slope_per_minute: float,
                               residual_std: float) -> dict:
    return {
        h: round(failure_probability_at(current_health, slope_per_minute, residual_std, h), 4)
        for h in config.FAILURE_PROB_HORIZONS_DAYS
    }
