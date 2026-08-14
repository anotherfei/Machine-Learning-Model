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
                "MAINTENANCE_TREND_DEBOUNCE_TICKS":10,
                "MAINTENANCE_TREND_RECOVERY_TICKS":5,
            })
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({"HEALTH_SENSITIVITY_STD":0})
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({"NEAR_MISS_TREND_WINDOW_HOURS":200})

    def test_extended_training_policy_validation(self):
        result=runtime_config.validate_training_config({
            "NEAR_MISS_REGRESSION_WINDOW_HOURS":2,
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
    def test_regression_floor_uses_calibrated_condition_risk(self):
        self.assertAlmostEqual(runtime_config.regression_risk_floor(60),.4)
        self.assertEqual(runtime_config.regression_risk_floor(100),.05)
        self.assertEqual(runtime_config.regression_risk_floor(0),.95)
        self.assertEqual(runtime_config.regression_risk_floor(float("nan")),.5)
    def test_health_critical_is_immediate(self):
        r=maintenance.recommend(10,90,{1:0},trend_trusted=False)
        self.assertEqual(r["level"],"CRITICAL"); self.assertEqual(r["trigger"],"health_threshold")
    def test_runtime_override_changes_policy_without_training(self):
        runtime_config.set_local({"FAILURE_HEALTH_THRESHOLD":5,"MAINTENANCE_HEALTH_INSPECT":15})
        r=maintenance.recommend(18,90,{1:0},trend_trusted=False)
        self.assertEqual(r["level"],"OK")

if __name__=='__main__': unittest.main()
