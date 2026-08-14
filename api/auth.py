from __future__ import annotations
import os
import time
from dataclasses import dataclass
import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, Response

COOKIE = "spindle_session"
ALGORITHM = "HS256"

def _secret() -> str:
    value = os.getenv("APP_SECRET_KEY")
    if not value:
        raise RuntimeError("APP_SECRET_KEY is not configured")
    return value

def _session_seconds() -> int:
    return int(os.getenv("APP_SESSION_SECONDS", "28800"))

@dataclass
class User:
    username: str
    role: str


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(password: str, password_hash: str) -> bool:
    try:
        return bcrypt.checkpw(password.encode(), password_hash.encode())
    except ValueError:
        return False


def issue_cookie(response: Response, username: str, role: str) -> None:
    now = int(time.time())
    session_seconds = _session_seconds()
    token = jwt.encode({"sub": username, "role": role, "iat": now, "exp": now + session_seconds}, _secret(), algorithm=ALGORITHM)
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=os.getenv("COOKIE_SECURE", "false").lower()=="true", max_age=session_seconds)


def session_user(token: str | None) -> User:
    """Validate a signed session token for HTTP or WebSocket transport."""
    if not token:
        raise HTTPException(401, "Login required")
    try:
        payload = jwt.decode(token, _secret(), algorithms=[ALGORITHM])
        username = payload.get("sub")
        role = payload.get("role")
        if not isinstance(username, str) or not username or role not in ("viewer", "admin"):
            raise HTTPException(401, "Invalid or expired session")
        return User(username, role)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(401, "Invalid or expired session")


def current_user(request: Request) -> User:
    return session_user(request.cookies.get(COOKIE))


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admin role required")
    return user
