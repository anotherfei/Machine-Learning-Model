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

    def test_incremental_trend_matches_array_fit_over_same_elapsed_window(self):
        window = trend_forecast.RollingTrendWindow(90.0)
        times = np.cumsum(np.resize(np.array([0.5, 1.0, 2.0, 0.75]), 180))
        values = 92.0 - 0.015 * times + 0.2 * np.sin(times / 7.0)
        for minute, health in zip(times, values):
            window.append(minute, health)

        cutoff = times[-1] - 90.0
        retained = times >= cutoff
        expected = trend_forecast.fit_trend(times[retained], values[retained])
        actual = window.fit()
        self.assertIsNotNone(expected)
        self.assertIsNotNone(actual)
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-9)
        self.assertEqual(
            window.slope_is_significant(actual[0], actual[2]),
            trend_forecast.slope_is_significant(times[retained], expected[0], expected[2]),
        )


if __name__ == "__main__":
    unittest.main()
