import unittest
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

import artifact_utils
import config
import feature_engineering
import operating_state
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

    def test_offline_motion_profile_can_learn_a_rare_stationary_regime(self):
        rng = np.random.default_rng(31)
        scores = np.concatenate([
            rng.normal(-0.2, 0.03, 19_970),
            rng.normal(-1.2, 0.02, 30),
        ])
        rng.shuffle(scores)
        detector = operating_state.OperatingStateDetector(history_rows=len(scores))
        for score in scores:
            detector.observe_activity_score(score)
        self.assertTrue(detector.fit_history(require_sustained_low=False))
        self.assertLess(detector.stop_threshold, detector.run_threshold)
        self.assertTrue(detector.last_fit_diagnostics["selected"]["accepted"])

    def test_commissioning_profile_accepts_strong_center_ratio_with_load_spread(self):
        rng = np.random.default_rng(47)
        scores = np.concatenate([
            rng.normal(-0.58, 0.28, 1500),
            rng.normal(0.45, 0.28, 500),
        ])
        rng.shuffle(scores)
        detector = operating_state.OperatingStateDetector(history_rows=len(scores))
        for score in scores:
            detector.observe_activity_score(score)
        self.assertTrue(detector.fit_history(require_sustained_low=False))
        selected = detector.last_fit_diagnostics["selected"]
        self.assertGreaterEqual(
            selected["separation_quality"],
            config.OPERATING_STATE_COMMISSIONING_MIN_SEPARATION_QUALITY,
        )
        self.assertGreater(selected["log_separation"], config.OPERATING_STATE_MIN_LOG_SEPARATION)

    def test_operator_confirmation_overrides_unknown_but_preserves_evidence(self):
        now = datetime.now(timezone.utc)
        detected = {
            "state": "UNKNOWN", "reason": "No separable regimes", "confidence": 0.0,
            "changed": False, "activity_score": -0.2, "low_motion": False,
            "stop_threshold": None, "run_threshold": None,
        }
        override = {
            "state": "RUNNING", "set_by": "operator", "set_at": now.isoformat(),
            "expires_at": (now + timedelta(hours=1)).isoformat(), "note": "Visual check",
        }
        effective = operating_state.apply_operator_override(detected, override, now)
        self.assertEqual(effective["state"], "RUNNING")
        self.assertEqual(effective["state_source"], "operator")
        self.assertEqual(effective["detected_state"], "UNKNOWN")
        self.assertFalse(effective["low_motion"])

    def test_operator_confirmation_expires_and_cannot_hide_sensor_fault(self):
        now = datetime.now(timezone.utc)
        expired = {
            "state": "STOPPED", "set_by": "operator",
            "set_at": (now - timedelta(hours=2)).isoformat(),
            "expires_at": (now - timedelta(hours=1)).isoformat(),
        }
        unknown = {
            "state": "UNKNOWN", "reason": "Unknown", "confidence": 0.0,
            "changed": False, "activity_score": 0.0, "low_motion": False,
            "stop_threshold": None, "run_threshold": None,
        }
        self.assertEqual(
            operating_state.apply_operator_override(unknown, expired, now)["state"],
            "UNKNOWN",
        )
        active = {**expired, "expires_at": (now + timedelta(hours=1)).isoformat()}
        fault = {**unknown, "state": "SENSOR_FAULT", "reason": "Invalid channel"}
        self.assertEqual(
            operating_state.apply_operator_override(fault, active, now)["state"],
            "SENSOR_FAULT",
        )


if __name__ == "__main__":
    unittest.main()
