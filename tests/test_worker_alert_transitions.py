import unittest

from maintenance import is_alert_escalation


class WorkerAlertTransitionTests(unittest.TestCase):
    def test_only_higher_severity_transitions_create_alerts(self):
        self.assertTrue(is_alert_escalation(None, "WARN"))
        self.assertTrue(is_alert_escalation("OK", "WARN"))
        self.assertTrue(is_alert_escalation("OK", "CRITICAL"))
        self.assertTrue(is_alert_escalation("WARN", "CRITICAL"))

        self.assertFalse(is_alert_escalation("WARN", "WARN"))
        self.assertFalse(is_alert_escalation("CRITICAL", "CRITICAL"))
        self.assertFalse(is_alert_escalation("CRITICAL", "WARN"))
        self.assertFalse(is_alert_escalation("WARN", "OK"))


if __name__ == "__main__":
    unittest.main()
