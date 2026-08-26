import unittest
import datetime as dt
from pathlib import Path

import simulation_service
import runtime_config


class HistoricalSimulationTests(unittest.TestCase):
    def setUp(self):
        runtime_config.set_local(runtime_config.defaults())

    def test_critical_event_requires_planning_warn_and_urgent_critical(self):
        onset = dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc)
        event_end = onset + dt.timedelta(hours=6)
        result = simulation_service._timing_assessment(
            "CRITICAL",
            onset,
            event_end,
            [onset - dt.timedelta(days=5)],
            [onset - dt.timedelta(hours=12)],
        )
        self.assertTrue(result["timing_pass"])
        self.assertTrue(result["planning_warning_pass"])
        self.assertTrue(result["urgent_critical_pass"])
        self.assertFalse(result["premature_critical"])
        self.assertEqual(result["first_warn_lead_hours"], 120)

    def test_premature_critical_fails_lead_time_policy(self):
        onset = dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc)
        result = simulation_service._timing_assessment(
            "CRITICAL",
            onset,
            onset + dt.timedelta(hours=6),
            [onset - dt.timedelta(days=5)],
            [onset - dt.timedelta(days=2), onset - dt.timedelta(hours=4)],
        )
        self.assertFalse(result["timing_pass"])
        self.assertTrue(result["premature_critical"])

    def test_ok_event_flags_any_lead_window_alert(self):
        onset = dt.datetime(2026, 1, 8, tzinfo=dt.timezone.utc)
        result = simulation_service._timing_assessment(
            "OK", onset, onset + dt.timedelta(hours=6),
            [onset - dt.timedelta(days=1)], []
        )
        self.assertFalse(result["timing_pass"])
        self.assertTrue(result["false_alert"])

    def test_replay_timestamps_are_always_aware_utc(self):
        naive_source = dt.datetime(2026, 8, 20, 9, 12)
        aware_event = dt.datetime(2026, 8, 20, 2, 12, tzinfo=dt.timezone.utc)
        normalized_source = simulation_service.utc_datetime(naive_source)
        normalized_event = simulation_service.utc_datetime(aware_event)
        self.assertEqual(normalized_source.tzinfo, dt.timezone.utc)
        self.assertEqual(normalized_event.tzinfo, dt.timezone.utc)
        # The comparison itself is the regression: it raised TypeError before
        # both source and event timestamps were normalized.
        self.assertIsInstance(normalized_source >= normalized_event, bool)

    def test_summary_separates_event_accuracy_coverage_and_timing(self):
        rows = [
            {"machine_id": "M-1", "result": {
                "evaluable": True, "match": True,
                "expected_status": "OK", "predicted_status": "OK",
                "expected_status_coverage": 0.9,
                "timing": {"timing_evaluable": True, "timing_pass": True},
            }},
            {"machine_id": "M-1", "result": {
                "evaluable": True, "match": False,
                "expected_status": "WARN", "predicted_status": "CRITICAL",
                "expected_status_coverage": 0.3,
                "timing": {
                    "timing_evaluable": True, "timing_pass": False,
                    "premature_critical": True,
                },
            }},
            {"machine_id": "M-2", "result": {
                "evaluable": False, "match": False,
                "expected_status": "OK", "predicted_status": "NO_DATA",
            }},
        ]
        summary = simulation_service.summarize(rows)
        self.assertEqual(summary["evaluated_ranges"], 2)
        self.assertEqual(summary["unscored_ranges"], 1)
        self.assertEqual(summary["accuracy"], 0.5)
        self.assertEqual(summary["target_accuracy"], 0.5)
        self.assertEqual(summary["event_accuracy"], 0.5)
        self.assertEqual(summary["mean_status_coverage"], 0.6)
        self.assertEqual(summary["timing_compliance"], 0.5)
        self.assertEqual(summary["premature_critical_cases"], 1)
        self.assertEqual(summary["confusion_matrix"]["WARN"]["CRITICAL"], 1)
        self.assertEqual(summary["per_machine"]["M-1"]["accuracy"], 0.5)

    def test_simulation_lists_and_results_share_one_database_table(self):
        root = Path(__file__).resolve().parents[1]
        schema = (root / "db_schema.py").read_text(encoding="utf-8")
        jobs = (root / "simulation_jobs.py").read_text(encoding="utf-8")
        api = (root / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn('CREATE TABLE IF NOT EXISTS "ML".simulation_runs', schema)
        self.assertIn("def save_draft", jobs)
        self.assertIn("def enqueue", jobs)
        self.assertNotIn("STORE_PATH", jobs)
        self.assertIn('"/api/simulation-templates"', api)

    def test_simulation_uses_selected_version_and_shared_batched_inference(self):
        root = Path(__file__).resolve().parents[1]
        contracts = (root / "api" / "contracts.py").read_text(encoding="utf-8")
        api = (root / "api" / "main.py").read_text(encoding="utf-8")
        service = (root / "simulation_service.py").read_text(encoding="utf-8")
        jobs = (root / "simulation_jobs.py").read_text(encoding="utf-8")
        self.assertIn("model_version: str | None = None", contracts)
        self.assertIn('requested_version=(body.model_version or "").strip()', api)
        self.assertIn("monitor.update_many", service)
        self.assertIn("predict_realtime.INFERENCE_BATCH_ROWS", service)
        self.assertIn('"rows_per_second"', service)
        self.assertIn("progress_callback=publish_progress", jobs)


if __name__ == "__main__":
    unittest.main()
