import unittest
import maintenance
import runtime_config

class RuntimePolicyTests(unittest.TestCase):
    def setUp(self): runtime_config.set_local(runtime_config.defaults())
    def test_threshold_validation(self):
        with self.assertRaises(ValueError):
            runtime_config.validate_thresholds({"MAINTENANCE_PROB_PLAN":.9,"MAINTENANCE_PROB_URGENT":.8})
    def test_health_critical_is_immediate(self):
        r=maintenance.recommend(10,90,{1:0},trend_trusted=False)
        self.assertEqual(r["level"],"CRITICAL"); self.assertEqual(r["trigger"],"health_threshold")
    def test_runtime_override_changes_policy_without_training(self):
        runtime_config.set_local({"FAILURE_HEALTH_THRESHOLD":5,"MAINTENANCE_HEALTH_INSPECT":15})
        r=maintenance.recommend(18,90,{1:0},trend_trusted=False)
        self.assertEqual(r["level"],"OK")

if __name__=='__main__': unittest.main()
