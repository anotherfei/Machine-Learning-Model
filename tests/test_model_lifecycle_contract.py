import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class ModelLifecycleContractTests(unittest.TestCase):
    def test_initial_training_writes_directly_to_version_registry(self):
        trainer = (ROOT / "train_isolation_forest.py").read_text(encoding="utf-8")
        self.assertIn("model_registry.bundle_path(version_id)", trainer)
        self.assertIn("directory=target", trainer)
        self.assertIn("register_initial_bundle", trainer)
        self.assertIn("initial_commissioning_forward_holdout", trainer)

    def test_model_names_are_labels_not_mutable_identifiers(self):
        registry = (ROOT / "model_registry.py").read_text(encoding="utf-8")
        self.assertIn("def rename_version", registry)
        self.assertIn("SET display_name=%s WHERE version_id=%s", registry)
        self.assertNotIn("UPDATE model_versions SET version_id=%s", registry)

    def test_switch_clears_only_unresolved_review_markers(self):
        registry = (ROOT / "model_registry.py").read_text(encoding="utf-8")
        self.assertIn("model_version<>%s AND alert_status='pending'", registry)
        self.assertIn("model_version<>%s AND near_miss_status='pending'", registry)
        self.assertNotIn("DELETE FROM spindle_predictions", registry)

    def test_saved_simulation_keeps_its_model_identity(self):
        registry = (ROOT / "model_registry.py").read_text(encoding="utf-8")
        self.assertIn("saved simulation", registry)
        self.assertIn("WHERE model_version=%s AND status<>'draft'", registry)

    def test_retraining_has_heldout_and_labelled_simulation_gates(self):
        service = (ROOT / "retrain_service.py").read_text(encoding="utf-8")
        self.assertIn('report["labelled_simulation_evaluation"]', service)
        self.assertIn('report["labelled_simulation_is_advisory"] = True', service)
        self.assertIn("accepted = True", service)


if __name__ == "__main__":
    unittest.main()
