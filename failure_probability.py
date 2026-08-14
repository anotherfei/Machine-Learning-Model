"""
Failure probability — turns the trend forecast's point estimate into a
probability distribution over when failure happens, instead of a single
number pretending to be certain.

Model: health evolves as Brownian motion with drift. The calculation uses the
closed-form first-passage probability for crossing the critical condition
boundary at any time up to the horizon, rather than merely asking whether the
terminal value is below the boundary. Its diffusion input is estimated from
detrended condition innovations by trend_forecast.estimate_diffusion(). This
is a model-based forecast risk, not an empirically calibrated guarantee; real
maintenance outcomes are still required to validate the numeric probabilities.
"""

import numpy as np
from scipy.special import log_ndtr
from scipy.stats import norm

import config
import runtime_config


def failure_probability_at(current_health: float, slope_per_minute: float,
                            diffusion_std_per_sqrt_minute: float, horizon_days: float) -> float:
    """First-passage P(condition crosses the critical boundary by horizon)."""
    horizon_minutes = horizon_days * 24 * 60
    failure_threshold = runtime_config.get(
        "FAILURE_HEALTH_THRESHOLD", config.FAILURE_HEALTH_THRESHOLD
    )
    distance = float(current_health) - float(failure_threshold)
    if distance <= 0:
        return 1.0
    if horizon_minutes <= 0:
        return 0.0

    # Transform declining condition into a positive degradation process that
    # starts at zero and hits `distance`. A negative condition slope therefore
    # becomes positive drift toward the boundary.
    drift = -float(slope_per_minute)
    sigma = max(float(diffusion_std_per_sqrt_minute), 0.0)
    if sigma <= 1e-12:
        return float(drift > 0 and drift * horizon_minutes >= distance)

    sigma_sqrt_t = sigma * np.sqrt(horizon_minutes)
    first = norm.cdf((drift * horizon_minutes - distance) / sigma_sqrt_t)
    second_log = (
        2.0 * drift * distance / (sigma * sigma)
        + log_ndtr((-drift * horizon_minutes - distance) / sigma_sqrt_t)
    )
    second = np.exp(min(float(second_log), 700.0))
    return float(np.clip(first + second, 0.0, 1.0))


def failure_probability_table(current_health: float, slope_per_minute: float,
                               diffusion_std_per_sqrt_minute: float) -> dict:
    return {
        h: round(failure_probability_at(
            current_health, slope_per_minute, diffusion_std_per_sqrt_minute, h,
        ), 4)
        for h in config.FAILURE_PROB_HORIZONS_DAYS
    }
