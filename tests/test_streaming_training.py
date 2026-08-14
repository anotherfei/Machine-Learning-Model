import unittest

import numpy as np
import pandas as pd

import artifact_utils
import config
import feature_engineering
from streaming_training import ForwardHoldoutReservoir, PriorityReservoir


class StreamingTrainingTests(unittest.TestCase):
    def test_priority_reservoir_is_independent_of_fetch_chunk_size(self):
        source = pd.DataFrame({"row_id": np.arange(1000), "value": np.arange(1000) * 0.5})
        one_chunk = PriorityReservoir(100, seed=17)
        one_chunk.offer(source)

        many_chunks = PriorityReservoir(100, seed=17)
        for start in range(0, len(source), 37):
            many_chunks.offer(source.iloc[start:start + 37])

        self.assertEqual(
            sorted(one_chunk.result()["row_id"].tolist()),
            sorted(many_chunks.result()["row_id"].tolist()),
        )

    def test_forward_holdout_always_preserves_newest_rows(self):
        sampler = ForwardHoldoutReservoir(
            capacity=100,
            holdout_fraction=artifact_utils.VALIDATION_HOLDOUT_FRACTION,
            minimum_holdout=artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS,
            seed=23,
        )
        source = pd.DataFrame({
            config.COL_TIMESTAMP: pd.date_range("2025-01-01", periods=1000, freq="s"),
            "value": np.arange(1000),
        })
        for start in range(0, len(source), 43):
            sampler.offer(source.iloc[start:start + 43])
        retained = sampler.result(config.COL_TIMESTAMP)
        newest = retained.tail(20)["value"].tolist()
        self.assertEqual(newest, list(range(980, 1000)))
        self.assertEqual(len(retained), 100)

    def test_chunked_features_match_full_trajectory(self):
        rows = 137
        source = pd.DataFrame({
            config.COL_TIMESTAMP: pd.date_range("2025-01-01", periods=rows, freq="s"),
            config.COL_A_RMS: 1.0 + np.sin(np.arange(rows) / 7),
            config.COL_V_RMS: 2.0 + np.cos(np.arange(rows) / 11),
            config.COL_A_PEAK: 4.0 + np.sin(np.arange(rows) / 5),
            config.COL_CREST_FACTOR: 3.0 + 0.1 * np.cos(np.arange(rows) / 3),
            config.COL_TEMPERATURE: 35.0 + np.arange(rows) * 0.01,
        })
        expected = feature_engineering.create_features(source, verbose=False)
        carry = None
        parts = []
        for start in range(0, rows, 17):
            featured, carry = feature_engineering.create_features_chunk(
                source.iloc[start:start + 17], carry
            )
            parts.append(featured)
        actual = pd.concat(parts, ignore_index=True)
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, rtol=1e-9, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
