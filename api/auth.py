from __future__ import annotations
import os
import time
from dataclasses import dataclass
import bcrypt
import jwt
from fastapi import Depends, HTTPException, Request, Response

SECRET = os.getenv("APP_SECRET_KEY", "change-me-before-production")
COOKIE = "spindle_session"
ALGORITHM = "HS256"
SESSION_SECONDS = int(os.getenv("APP_SESSION_SECONDS", "28800"))

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
    token = jwt.encode({"sub": username, "role": role, "iat": now, "exp": now + SESSION_SECONDS}, SECRET, algorithm=ALGORITHM)
    response.set_cookie(COOKIE, token, httponly=True, samesite="strict", secure=os.getenv("COOKIE_SECURE", "false").lower()=="true", max_age=SESSION_SECONDS)


def current_user(request: Request) -> User:
    token = request.cookies.get(COOKIE)
    if not token:
        raise HTTPException(401, "Login required")
    try:
        payload = jwt.decode(token, SECRET, algorithms=[ALGORITHM])
        return User(payload["sub"], payload["role"])
    except Exception:
        raise HTTPException(401, "Invalid or expired session")


def require_admin(user: User = Depends(current_user)) -> User:
    if user.role != "admin":
        raise HTTPException(403, "Admin role required")
    return user
