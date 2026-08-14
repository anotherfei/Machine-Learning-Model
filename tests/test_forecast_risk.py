import unittest

import numpy as np

import failure_probability
import runtime_config
import trend_forecast


class ForecastRiskTests(unittest.TestCase):
    def setUp(self):
        runtime_config.set_local(runtime_config.defaults())

    def test_boundary_is_already_crossed(self):
        risk = failure_probability.failure_probability_at(10, 0, 0.1, 1)
        self.assertEqual(risk, 1.0)

    def test_deterministic_decline_crosses_within_horizon(self):
        risk = failure_probability.failure_probability_at(40, -1 / 60, 0, 1)
        self.assertEqual(risk, 1.0)

    def test_deterministic_improvement_does_not_cross(self):
        risk = failure_probability.failure_probability_at(40, 1 / 60, 0, 1)
        self.assertEqual(risk, 0.0)

    def test_longer_horizon_does_not_reduce_first_passage_risk(self):
        short = failure_probability.failure_probability_at(60, -0.005, 0.05, 0.25)
        long = failure_probability.failure_probability_at(60, -0.005, 0.05, 1)
        self.assertGreaterEqual(long, short)

    def test_diffusion_uses_actual_time_intervals(self):
        times = np.array([0.0, 5.0, 20.0, 45.0])
        values = np.array([90.0, 89.7, 89.1, 88.2])
        slope, intercept, _ = trend_forecast.fit_trend(times, values)
        diffusion = trend_forecast.estimate_diffusion(times, values, slope, intercept)
        self.assertTrue(np.isfinite(diffusion))
        self.assertGreater(diffusion, 0)


if __name__ == "__main__":
    unittest.main()
