import unittest

import pandas as pd

import config
from train_isolation_forest import _balanced_reference_pool, _split_validation_holdout


def _frame(machine_offset: int, rows: int) -> pd.DataFrame:
    return pd.DataFrame({
        config.COL_TIMESTAMP: pd.date_range(
            "2026-01-01", periods=rows, freq="min", tz="UTC"
        ),
        "feature_a": [machine_offset + value for value in range(rows)],
        "feature_b": [machine_offset + value / 10 for value in range(rows)],
    })


class CommissioningHoldoutTests(unittest.TestCase):
    def test_balanced_split_keeps_equal_forward_holdout(self):
        balanced, _, rows_per_machine = _balanced_reference_pool({
            "MACHINE-001": _frame(0, 120),
            "MACHINE-002": _frame(1000, 100),
        })
        training, validation = _split_validation_holdout(
            balanced, ["feature_a", "feature_b"]
        )

        self.assertEqual(rows_per_machine, 100)
        self.assertEqual({key: len(value) for key, value in training.items()}, {
            "MACHINE-001": 80,
            "MACHINE-002": 80,
        })
        self.assertEqual({key: len(value) for key, value in validation.items()}, {
            "MACHINE-001": 20,
            "MACHINE-002": 20,
        })
        for machine_id in training:
            self.assertLess(
                training[machine_id][config.COL_TIMESTAMP].max(),
                validation[machine_id][config.COL_TIMESTAMP].min(),
            )

    def test_too_small_reference_fails_before_fitting(self):
        with self.assertRaisesRegex(ValueError, "balanced healthy feature rows"):
            _split_validation_holdout(
                {"MACHINE-001": _frame(0, 30)},
                ["feature_a", "feature_b"],
            )


if __name__ == "__main__":
    unittest.main()
