import os
import tempfile
import unittest
from pathlib import Path

import env_manager


class EnvManagerTests(unittest.TestCase):
    def test_mask_constant_is_not_plaintext(self):
        self.assertNotEqual(env_manager.MASK, "")

    def test_required_fields_declared(self):
        self.assertIn("PG_PASSWORD", env_manager.REQUIRED)

    def test_all_credentials_are_masked(self):
        self.assertTrue({"PG_PASSWORD","APP_SECRET_KEY","BOOTSTRAP_ADMIN_PASSWORD"}.issubset(env_manager.SECRET_KEYS))

    def test_session_duration_is_web_editable(self):
        self.assertIn("APP_SESSION_SECONDS", env_manager.ALLOWED)

    def test_public_placeholder_is_not_a_secure_signing_key(self):
        self.assertFalse(env_manager.app_secret_is_secure("change-me-before-production"))
        self.assertTrue(env_manager.app_secret_is_secure("x" * 32))

    def test_bootstrap_default_is_not_a_secure_password(self):
        self.assertFalse(env_manager.bootstrap_password_is_secure("change-me-on-first-deployment"))
        self.assertTrue(env_manager.bootstrap_password_is_secure("a-unique-password-123"))


class LocalEnvironmentRefreshTests(unittest.TestCase):
    def test_apply_to_process_environment_uses_file_values(self):
        old_path = env_manager.ENV_PATH
        old_host = os.environ.get("PG_HOST")
        try:
            with tempfile.TemporaryDirectory() as td:
                env_manager.ENV_PATH = Path(td) / ".env"
                env_manager.ENV_PATH.write_text(
                    "PG_HOST=new-local-host\nPG_DATABASE=db\nPG_USER=u\nPG_PASSWORD=p\nPG_TABLE=t\n",
                    encoding="utf-8",
                )
                os.environ["PG_HOST"] = "stale-shell-host"
                env_manager.apply_to_process_environment()
                self.assertEqual(os.environ["PG_HOST"], "new-local-host")
        finally:
            env_manager.ENV_PATH = old_path
            if old_host is None:
                os.environ.pop("PG_HOST", None)
            else:
                os.environ["PG_HOST"] = old_host


if __name__ == "__main__":
    unittest.main()
