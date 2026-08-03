"""
Kalman filter for denoising the raw health-percentage signal into a
smoothed "estimated health state".

Deliberately a simple constant-level (random-walk) filter — ONE state
(level), no velocity/trend component. Trend estimation is a separate,
independently swappable stage (trend_forecast.py). Folding both into one
Kalman filter was tried in an earlier version of this pipeline and caused
problems: a velocity state estimated from noisy per-tick measurements
produced wildly unstable remaining-life extrapolations, because a small
noisy velocity blows up when divided into a "time to threshold" number.
Keeping this filter to pure denoising, with a dedicated regression-based
module handling extrapolation over a longer lookback window, avoids that
failure mode entirely and matches the modular architecture described in
README.md — swap trend_forecast.py without ever touching this file.
"""

import config


class HealthKalmanFilter:
    def __init__(self, initial_level: float):
        params = config.KALMAN_PARAMS
        self.level = initial_level
        self.variance = params["measurement_var"]  # start with measurement uncertainty
        self.process_var = params["process_var"]
        self.measurement_var = params["measurement_var"]

    def update(self, measurement: float) -> float:
        # Predict
        pred_level = self.level
        pred_variance = self.variance + self.process_var

        # Update
        kalman_gain = pred_variance / (pred_variance + self.measurement_var)
        self.level = pred_level + kalman_gain * (measurement - pred_level)
        self.variance = (1 - kalman_gain) * pred_variance

        return self.level
