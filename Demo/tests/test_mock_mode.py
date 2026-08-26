from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[2]


class MockModeTests(unittest.TestCase):
    def test_mock_mode_is_sqlite_and_skips_production_worker(self):
        launcher = (ROOT / "local_launcher.py").read_text(encoding="utf-8")
        mock_api = (ROOT / "Demo" / "mock_main.py").read_text(encoding="utf-8")
        start = (ROOT / "start_project.ps1").read_text(encoding="utf-8")
        self.assertIn("--mock", launcher)
        self.assertIn("Demo.mock_main:app", launcher)
        self.assertIn("sqlite3", mock_api)
        self.assertIn("[switch]$Mock", start)
        self.assertIn("if not args.mock", launcher)
        db_source = (ROOT / "db.py").read_text(encoding="utf-8")
        self.assertNotIn("def parse_args", db_source)
        self.assertNotIn("import argparse", db_source)

    def test_mock_api_keeps_shared_review_contracts(self):
        mock_api = (ROOT / "Demo" / "mock_main.py").read_text(encoding="utf-8")
        production_api = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        contracts = (ROOT / "api" / "contracts.py").read_text(encoding="utf-8")
        self.assertIn("from api.contracts import", mock_api)
        self.assertIn("from api.contracts import", production_api)
        self.assertIn("class ThresholdBody", contracts)
        self.assertNotIn("class ThresholdBody", mock_api)
        self.assertNotIn("class ThresholdBody", production_api)
        self.assertIn('"/api/near-miss/{prediction_id}/review"', mock_api)
        self.assertIn("Near-miss record not found", mock_api)
        self.assertIn("decision must be acknowledged or flagged", mock_api)
        self.assertNotIn('"/api/near-miss/{prediction_id}/promote"', mock_api)
        self.assertIn("alert_status", mock_api)
        self.assertIn("near_miss_status", mock_api)

    def test_removed_legacy_name_and_container_files(self):
        forbidden = ("Project" + " F", "project" + "_f", "PROJECT" + "_F")
        for path in ROOT.rglob("*"):
            if not path.is_file() or "node_modules" in path.parts or path.suffix in {".pyc", ".db"}:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            for token in forbidden:
                self.assertNotIn(token, text, f"{token!r} remains in {path.relative_to(ROOT)}")
        self.assertFalse((ROOT / "Dockerfile").exists())
        self.assertFalse((ROOT / "docker-compose.yml").exists())
        self.assertFalse((ROOT / "frontend" / "Dockerfile").exists())


if __name__ == "__main__":
    unittest.main()
