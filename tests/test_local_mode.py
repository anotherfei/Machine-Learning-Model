from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class LocalModeTests(unittest.TestCase):
    def test_vite_proxy_uses_local_api(self):
        text = (ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
        self.assertIn("127.0.0.1", text)

    def test_local_launchers_exist(self):
        for name in ("worker_supervisor.py", "local_launcher.py", "setup_local.ps1", "start_project.ps1"):
            self.assertTrue((ROOT / name).exists(), name)

    def test_default_frontend_origin_matches_vite(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("FRONTEND_ORIGIN=http://localhost:5173", text)

    def test_worker_restart_contract_is_supervised(self):
        worker = (ROOT / "worker.py").read_text(encoding="utf-8")
        supervisor = (ROOT / "worker_supervisor.py").read_text(encoding="utf-8")
        self.assertIn("return 75", worker)
        self.assertIn("RESTART_CODE = 75", supervisor)

    def test_near_miss_has_own_review_workflow(self):
        real_api = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        frontend = (ROOT / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
        schema = (ROOT / "db_schema.py").read_text(encoding="utf-8")
        self.assertIn("near_miss_reviews", schema)
        self.assertIn("CHECK (status IN ('pending','acknowledged','flagged'))", schema)
        self.assertIn('"/api/near-miss/{prediction_id}/review"', real_api)
        self.assertIn("Near-miss record not found", real_api)
        self.assertIn("decision must be acknowledged or flagged", real_api)
        self.assertNotIn('"/api/near-miss/{prediction_id}/promote"', real_api)
        self.assertIn("/api/near-miss/${selected.id}/review", frontend)
        self.assertIn("Acknowledge", frontend)
        self.assertIn("Flag for follow-up", frontend)

    def test_history_shows_downstream_review_outcome(self):
        real_api = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        frontend = (ROOT / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
        self.assertIn("alert_status", real_api)
        self.assertIn("near_miss_status", real_api)
        self.assertIn("historyOutcome", frontend)
        self.assertIn("Alert outcome", frontend)
        self.assertIn("Near-miss outcome", frontend)


if __name__ == "__main__":
    unittest.main()

class FrontendStartupTests(unittest.TestCase):
    def test_frontend_never_intentionally_returns_blank_during_auth_check(self):
        text = (ROOT / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
        self.assertNotIn("if(me===undefined)return null", text.replace(" ", ""))
        self.assertIn("Connecting to the local API", text)
        self.assertIn("AbortController", text)

    def test_launcher_waits_for_api_before_frontend(self):
        text = (ROOT / "local_launcher.py").read_text(encoding="utf-8")
        self.assertIn("/api/health", text)
        self.assertIn("_wait_for_api", text)

    def test_frontend_dependencies_are_pinned_and_types_present(self):
        import json
        package = json.loads((ROOT / "frontend" / "package.json").read_text(encoding="utf-8"))
        self.assertNotEqual(package["dependencies"]["react"], "latest")
        self.assertIn("@types/react", package["devDependencies"])
        self.assertIn("@types/react-dom", package["devDependencies"])
