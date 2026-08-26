from pathlib import Path
import unittest
from datetime import timedelta

ROOT = Path(__file__).resolve().parents[1]


class LocalModeTests(unittest.TestCase):
    def test_active_model_uses_an_atomic_directory_pointer(self):
        registry = (ROOT / "model_registry.py").read_text(encoding="utf-8")
        artifacts = (ROOT / "artifact_utils.py").read_text(encoding="utf-8")
        api = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        self.assertIn('ACTIVE_POINTER = os.path.join(config.ARTIFACTS_DIR, "active.json")', registry)
        self.assertIn("os.replace(temporary, ACTIVE_POINTER)", registry)
        self.assertIn('"bundle_dir": (Path("versions") / version_id).as_posix()', registry)
        self.assertNotIn("def install_bundle", registry)
        self.assertIn("active_bundle_path(required=False)", artifacts)
        self.assertIn("active_id=model_registry.active_version()", api)

    def test_inference_has_one_production_polling_owner(self):
        engine = (ROOT / "predict_realtime.py").read_text(encoding="utf-8")
        worker = (ROOT / "worker.py").read_text(encoding="utf-8")
        self.assertNotIn("def read_sensor", engine)
        self.assertNotIn('if __name__ == "__main__"', engine)
        self.assertNotIn("def reload_model", engine)
        self.assertIn("bundle=predict_realtime.ModelBundle()", worker)
        self.assertIn("bundle.create_monitor(machine_id)", worker)
        self.assertIn("def main()", worker)
        self.assertIn("fetch_new_rows", worker)

    def test_live_timestamp_parser_accepts_variable_fraction_and_normalizes_utc(self):
        from api.main import _comparison_timestamp

        variable_fraction = _comparison_timestamp("2026-08-19T15:32:07.62175+07:00")
        explicit_utc = _comparison_timestamp("2026-08-19T08:32:07.621750Z")
        naive = _comparison_timestamp("2026-08-19T15:32:07.62175")
        self.assertIsNotNone(variable_fraction)
        self.assertIsNotNone(naive)
        self.assertEqual(variable_fraction.microsecond, 621750)
        self.assertEqual(variable_fraction.utcoffset(), timedelta(0))
        self.assertEqual(naive.utcoffset(), timedelta(0))
        self.assertEqual(variable_fraction, explicit_utc)

    def test_vite_proxy_uses_local_api(self):
        text = (ROOT / "frontend" / "vite.config.ts").read_text(encoding="utf-8")
        self.assertIn("127.0.0.1", text)

    def test_local_launchers_exist(self):
        for name in ("local_launcher.py", "setup_local.ps1", "start_project.ps1"):
            self.assertTrue((ROOT / name).exists(), name)

    def test_optional_startup_backfill_hands_state_into_live_worker(self):
        powershell = (ROOT / "start_project.ps1").read_text(encoding="utf-8")
        launcher = (ROOT / "local_launcher.py").read_text(encoding="utf-8")
        worker = (ROOT / "worker.py").read_text(encoding="utf-8")
        self.assertIn("BackfillDays", powershell)
        self.assertIn('"--backfill-days"', launcher)
        self.assertIn('"--catch-up-days"', launcher)
        self.assertIn('"--catch-up-days"', worker)
        self.assertIn('"--schema-ready"',launcher)
        self.assertIn('"--schema-ready"',worker)
        self.assertIn("_run_sequential_catchup", worker)
        self.assertIn('processes.append(_start("worker",worker_command,ROOT))', launcher)
        self.assertIn("_publish_catchup_launch",launcher)
        self.assertIn('"status":"launching"',launcher)
        self.assertIn('"status":"connecting"',worker)
        self.assertIn("connect_timeout=10",(ROOT/"db.py").read_text(encoding="utf-8"))
        self.assertIn("TRAINING_DB_STATEMENT_TIMEOUT_MS",worker)
        self.assertIn("API and frontend remain",launcher)
        self.assertNotIn('"history backfill"', launcher)
        self.assertNotIn("low_priority=True", launcher)
        self.assertIn('[sys.executable,"worker.py","--schema-ready"]',launcher)
        backfill = (ROOT / "backfill.py").read_text(encoding="utf-8")
        self.assertIn("iter_row_chunks_between", backfill)
        self.assertIn("execute_values", backfill)
        self.assertIn("skip_existing=True", worker)
        self.assertIn("return_runtime=True", worker)
        self.assertIn('"state_detectors":runtime_detectors',backfill)
        self.assertIn('"last_seen":last_seen',backfill)
        self.assertIn('"operating_snapshots":runtime_snapshots',backfill)
        self.assertIn("_persist_catchup_operating_states",worker)
        self.assertIn("continuity_guard",backfill)
        self.assertIn('phase="motion_profile"',backfill)
        self.assertIn("Profiling motion regimes",frontend)
        self.assertIn("_catchup_context_revision",worker)
        self.assertIn("runtime_config.THRESHOLD_KEYS",worker)
        self.assertIn("CatchupContextChanged",worker)
        self.assertIn("DO NOTHING", backfill)
        self.assertNotIn("class _BackfillProgress", backfill)
        self.assertNotIn('print(f"\\r', backfill)
        self.assertNotIn("cur.fetchall()", backfill)

    def test_backfill_progress_is_global_and_api_backed(self):
        api_text=(ROOT/"api"/"main.py").read_text(encoding="utf-8")
        mock_text=(ROOT/"Demo"/"mock_main.py").read_text(encoding="utf-8")
        frontend=(ROOT/"frontend"/"src"/"main.tsx").read_text(encoding="utf-8")
        schema=(ROOT/"db_schema.py").read_text(encoding="utf-8")
        self.assertIn('"/api/backfill/status"',api_text)
        self.assertIn('"/api/backfill/status"',mock_text)
        self.assertIn('"/api/fleet/latest"',api_text)
        self.assertIn('"/api/fleet/latest"',mock_text)
        self.assertIn("BackfillProgressPopup",frontend)
        self.assertIn("backfillStatus",frontend)
        self.assertIn("Preparing historical catch-up",frontend)
        self.assertIn("/api/fleet/latest?machine_ids=",frontend)
        self.assertNotIn("Promise.allSettled(machines.map",frontend)
        self.assertIn("write_runtime_status",worker)
        self.assertIn("read_runtime_status",api_text)
        self.assertIn("backfill_status.json",(ROOT/"backfill.py").read_text(encoding="utf-8"))
        self.assertIn("'backfill'",schema)
        self.assertIn("previous application session ended before backfill completed",api_text)

    def test_fleet_summary_separates_monitoring_from_confirmed_motion(self):
        frontend=(ROOT/"frontend"/"src"/"main.tsx").read_text(encoding="utf-8")
        api_text=(ROOT/"api"/"main.py").read_text(encoding="utf-8")
        self.assertIn("Actively monitored",frontend)
        self.assertIn("confirmed running",frontend)
        self.assertIn("ML active · motion unconfirmed",frontend)
        self.assertIn("locked until sequential catch-up finishes",api_text)
        self.assertIn("'preparing','running','restarting','draining'",api_text)

    def test_manual_motion_confirmation_is_bounded_audited_and_worker_applied(self):
        api_text=(ROOT/"api"/"main.py").read_text(encoding="utf-8")
        worker=(ROOT/"worker.py").read_text(encoding="utf-8")
        frontend=(ROOT/"frontend"/"src"/"main.tsx").read_text(encoding="utf-8")
        schema=(ROOT/"db_schema.py").read_text(encoding="utf-8")
        self.assertIn('/api/machines/{machine_id}/operating-override',api_text)
        self.assertIn("Depends(require_admin)",api_text)
        self.assertIn("operating_override_changed",worker)
        self.assertIn("apply_operator_override",worker)
        self.assertIn("Confirm manually",frontend)
        self.assertIn("Confirmation duration",frontend)
        self.assertIn("operating_override",schema)

    def test_default_frontend_origin_matches_vite(self):
        text = (ROOT / ".env.example").read_text(encoding="utf-8")
        self.assertIn("FRONTEND_ORIGIN=http://localhost:5173", text)

    def test_worker_restart_contract_is_supervised(self):
        worker = (ROOT / "worker.py").read_text(encoding="utf-8")
        launcher = (ROOT / "local_launcher.py").read_text(encoding="utf-8")
        self.assertIn("return 75", worker)
        self.assertIn("WORKER_RESTART_CODE = 75", launcher)
        self.assertIn('[sys.executable, "worker.py"]', launcher)

    def test_near_miss_has_own_review_workflow(self):
        real_api = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
        frontend = (ROOT / "frontend" / "src" / "main.tsx").read_text(encoding="utf-8")
        schema = (ROOT / "db_schema.py").read_text(encoding="utf-8")
        self.assertIn("near_miss_status", schema)
        self.assertIn("CHECK (near_miss_status IN ('pending','acknowledged','flagged'))", schema)
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
