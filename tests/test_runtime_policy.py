import unittest
import maintenance
import runtime_config

class RuntimePolicyTests(unittest.TestCase):
    def setUp(self): runtime_config.set_local(runtime_config.defaults())
    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({"MAINTENANCE_PROB_PLAN":.9,"MAINTENANCE_PROB_URGENT":.8})
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({"MAINTENANCE_HORIZON_DAYS":2})
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({
                "MAINTENANCE_HORIZON_DAYS": 7,
                "MAINTENANCE_URGENT_HORIZON_DAYS": 7,
            })
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({
                "MAINTENANCE_HORIZON_DAYS": 1,
                "MAINTENANCE_URGENT_HORIZON_DAYS": 1,
            })
        normalized = runtime_config.normalize_loaded_values({
            "MAINTENANCE_HORIZON_DAYS": 7,
            "MAINTENANCE_URGENT_HORIZON_DAYS": 7,
        })
        self.assertEqual(normalized["MAINTENANCE_URGENT_HORIZON_DAYS"], 1)
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({
                "MAINTENANCE_CRITICAL_CONFIRM_MINUTES":4,
                "MAINTENANCE_WARN_CONFIRM_MINUTES":5,
            })
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({"HEALTH_SENSITIVITY_STD":0})
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({"NEAR_MISS_TREND_WINDOW_HOURS":200})

    def test_extended_training_policy_validation(self):
        result=runtime_config.validate_training_config({
            "RETRAIN_CHECK_INTERVAL_MINUTES":30,
        })
        self.assertEqual(result["RETRAIN_CHECK_INTERVAL_MINUTES"],30)
        with self.assertRaises(ValueError):
            runtime_config.validate_training_config({"RETRAIN_CHECK_INTERVAL_MINUTES":1})
        with self.assertRaises(ValueError):
            runtime_config.validate_training_config({"RETRAIN_BATCH_SIZE":1.5})
        with self.assertRaises(ValueError):
            runtime_config.validate_training_config({"REFERENCE_WINDOW_MONTHS":1,"RETRAIN_TIME_CAP_DAYS":30})
        with self.assertRaises(ValueError):
            runtime_config.validate_training_config({"RETRAIN_MAX_FP_RATE_INCREASE":.2})
    def test_context_override_does_not_replace_process_policy(self):
        runtime_config.set_local({"FAILURE_HEALTH_THRESHOLD":20})
        with runtime_config.override({"FAILURE_HEALTH_THRESHOLD":12}):
            self.assertEqual(runtime_config.get("FAILURE_HEALTH_THRESHOLD"),12)
        self.assertEqual(runtime_config.get("FAILURE_HEALTH_THRESHOLD"),20)
    def test_health_critical_is_immediate(self):
        r=maintenance.recommend(10,90,{1:0},trend_trusted=False)
        self.assertEqual(r["level"],"CRITICAL"); self.assertEqual(r["trigger"],"health_threshold")
    def test_runtime_override_changes_policy_without_training(self):
        runtime_config.set_local({"FAILURE_HEALTH_THRESHOLD":5,"MAINTENANCE_HEALTH_INSPECT":15})
        r=maintenance.recommend(18,90,{1:0},trend_trusted=False)
        self.assertEqual(r["level"],"OK")

    def test_status_stabilization_uses_elapsed_time(self):
        debouncer = maintenance.MaintenanceDebouncer()
        runtime_config.set_local({
            "MAINTENANCE_CRITICAL_CONFIRM_MINUTES": 10,
            "MAINTENANCE_WARN_CONFIRM_MINUTES": 5,
            "MAINTENANCE_RECOVERY_MINUTES": 10,
            "SOURCE_STALE_SECONDS": 900,
        })
        self.assertEqual(debouncer.evaluate(25, 90, {1: 0, 7: 0}, False, 0)["level"], "OK")
        self.assertEqual(debouncer.evaluate(25, 90, {1: 0, 7: 0}, False, 4.9)["level"], "OK")
        self.assertEqual(debouncer.evaluate(25, 90, {1: 0, 7: 0}, False, 5)["level"], "WARN")
        self.assertEqual(debouncer.evaluate(90, 90, {1: 0, 7: 0}, False, 14.9)["level"], "WARN")
        self.assertEqual(debouncer.evaluate(90, 90, {1: 0, 7: 0}, False, 24.9)["level"], "OK")

    def test_critical_evidence_advances_through_warning_stage(self):
        debouncer = maintenance.MaintenanceDebouncer()
        runtime_config.set_local({
            "MAINTENANCE_WARN_CONFIRM_MINUTES": 5,
            "MAINTENANCE_CRITICAL_CONFIRM_MINUTES": 10,
            "MAINTENANCE_RECOVERY_MINUTES": 10,
            "SOURCE_STALE_SECONDS": 900,
        })
        self.assertEqual(debouncer.evaluate(10, 1, {1: 1}, True, 0)["level"], "OK")
        warning = debouncer.evaluate(10, 1, {1: 1}, True, 5)
        self.assertEqual(warning["level"], "WARN")
        self.assertTrue(warning["stabilizing"])
        self.assertEqual(warning["candidate_level"], "CRITICAL")
        self.assertEqual(debouncer.evaluate(10, 1, {1: 1}, True, 10)["level"], "CRITICAL")

    def test_missing_source_gap_does_not_confirm_candidate(self):
        debouncer = maintenance.MaintenanceDebouncer()
        runtime_config.set_local({
            "MAINTENANCE_WARN_CONFIRM_MINUTES": 5,
            "SOURCE_STALE_SECONDS": 180,
        })
        self.assertEqual(debouncer.evaluate(25, 90, {1: 0, 7: 0}, False, 0)["level"], "OK")
        self.assertEqual(debouncer.evaluate(25, 90, {1: 0, 7: 0}, False, 10)["level"], "OK")
        self.assertEqual(debouncer.evaluate(25, 90, {1: 0, 7: 0}, False, 12)["level"], "OK")
        self.assertEqual(debouncer.evaluate(25, 90, {1: 0, 7: 0}, False, 15)["level"], "WARN")

    def test_weekly_risk_warns_without_making_near_term_critical(self):
        result = maintenance.recommend(80, 20, {1: .2, 7: .9}, trend_trusted=True)
        self.assertEqual(result["level"], "WARN")

if __name__=='__main__': unittest.main()
