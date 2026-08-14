import os
import time
import unittest
from unittest.mock import patch

import jwt
from fastapi import HTTPException

from api.auth import ALGORITHM, session_user


class SessionPolicyTests(unittest.TestCase):
    def setUp(self):
        self.secret = "test-only-session-secret-that-is-long-enough"

    def token(self, **overrides):
        payload = {
            "sub": "operator",
            "role": "viewer",
            "iat": int(time.time()),
            "exp": int(time.time()) + 60,
            **overrides,
        }
        return jwt.encode(payload, self.secret, algorithm=ALGORITHM)

    def test_valid_session_claims(self):
        with patch.dict(os.environ, {"APP_SECRET_KEY": self.secret}):
            user = session_user(self.token())
        self.assertEqual(user.username, "operator")
        self.assertEqual(user.role, "viewer")

    def test_missing_or_invalid_role_is_rejected(self):
        with patch.dict(os.environ, {"APP_SECRET_KEY": self.secret}):
            with self.assertRaises(HTTPException):
                session_user(None)
            with self.assertRaises(HTTPException):
                session_user(self.token(role="owner"))

    def test_expired_session_is_rejected(self):
        with patch.dict(os.environ, {"APP_SECRET_KEY": self.secret}):
            with self.assertRaises(HTTPException):
                session_user(self.token(exp=int(time.time()) - 1))


if __name__ == "__main__":
    unittest.main()
