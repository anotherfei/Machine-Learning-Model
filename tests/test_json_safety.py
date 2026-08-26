import unittest

import numpy as np

import artifact_utils


class JsonSafetyTests(unittest.TestCase):
    def test_nested_numpy_scalars_become_native_python_values(self):
        converted = artifact_utils.to_json_safe({
            "trend_slope_per_day": np.float64(-0.125),
            "remaining_days": np.int64(12),
            "failure_probability": {1: np.float32(0.25)},
            "contributors": [("velocity", np.float64(2.5), np.bool_(True))],
        })

        self.assertIs(type(converted["trend_slope_per_day"]), float)
        self.assertIs(type(converted["remaining_days"]), int)
        self.assertIs(type(converted["failure_probability"]["1"]), float)
        self.assertIs(type(converted["contributors"][0][1]), float)
        self.assertIs(type(converted["contributors"][0][2]), bool)


if __name__ == "__main__":
    unittest.main()
