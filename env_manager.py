"""Atomic, masked .env read/write with an allow-list."""
from __future__ import annotations
import os
import secrets
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"
MASK = "••••••••"
REQUIRED = ("PG_HOST", "PG_DATABASE", "PG_USER", "PG_PASSWORD", "PG_TABLE")
ALLOWED = set(REQUIRED + ("PG_PORT", "PG_COL_TIMESTAMP", "PG_COL_MACHINE_ID", "DEFAULT_MACHINE_ID", "PG_COL_A_RMS_MPS2", "PG_COL_V_RMS_MMS", "PG_COL_A_PEAK_MPS2", "PG_COL_CREST_FACTOR", "PG_COL_TEMPERATURE_C", "APP_SECRET_KEY", "APP_SESSION_SECONDS", "FRONTEND_ORIGIN", "COOKIE_SECURE", "BOOTSTRAP_ADMIN_USER", "BOOTSTRAP_ADMIN_PASSWORD"))
SECRET_KEYS = {"PG_PASSWORD", "APP_SECRET_KEY", "BOOTSTRAP_ADMIN_PASSWORD"}
INSECURE_APP_SECRETS = {"change-me-before-production", "replace-with-a-long-random-secret"}
INSECURE_BOOTSTRAP_PASSWORDS = {"change-me-on-first-deployment", "replace-with-a-strong-unique-password"}


def app_secret_is_secure(value: str | None) -> bool:
    value = (value or "").strip()
    return len(value) >= 32 and value not in INSECURE_APP_SECRETS


def bootstrap_password_is_secure(value: str | None) -> bool:
    value = value or ""
    return len(value) >= 12 and value not in INSECURE_BOOTSTRAP_PASSWORDS


def _parse() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}
    out = {}
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        out[k.strip()] = v.strip()
    return out


def read_masked() -> dict[str, str]:
    values = _parse()
    values.setdefault("APP_SESSION_SECONDS", os.getenv("APP_SESSION_SECONDS", "28800"))
    ordered = list(REQUIRED) + [k for k in sorted(ALLOWED) if k not in REQUIRED]
    return {k: (MASK if k in SECRET_KEYS and values.get(k) else values.get(k, "")) for k in ordered}


def write(values: dict[str, str]) -> list[str]:
    current = _parse()
    changed = []
    for key, value in values.items():
        if key not in ALLOWED:
            raise ValueError(f"Unsupported .env key: {key}")
        if key in SECRET_KEYS and value == MASK:
            continue
        value = str(value).replace("\n", "").replace("\r", "")
        if key == "APP_SECRET_KEY" and not app_secret_is_secure(value):
            raise ValueError("APP_SECRET_KEY must be a unique secret of at least 32 characters")
        if key == "BOOTSTRAP_ADMIN_PASSWORD" and value and not bootstrap_password_is_secure(value):
            raise ValueError("BOOTSTRAP_ADMIN_PASSWORD must be a unique password of at least 12 characters")
        if key == "APP_SESSION_SECONDS":
            try:
                seconds = int(value)
            except ValueError as exc:
                raise ValueError("APP_SESSION_SECONDS must be a whole number") from exc
            if not 300 <= seconds <= 604800:
                raise ValueError("APP_SESSION_SECONDS must be between 300 and 604800")
            value = str(seconds)
        if current.get(key) != value:
            current[key] = value
            changed.append(key)
    missing = [k for k in REQUIRED if not current.get(k)]
    if missing:
        raise ValueError(f"Missing required .env keys: {missing}")
    text = "\n".join(f"{k}={current[k]}" for k in sorted(current) if k in ALLOWED) + "\n"
    tmp = ENV_PATH.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, ENV_PATH)
    return changed


def ensure_secure_app_secret() -> bool:
    """Generate and persist a signing key when production still has a placeholder.

    Returns True when a key was generated. Existing secure keys are preserved.
    """
    current = _parse()
    value = current.get("APP_SECRET_KEY") or os.getenv("APP_SECRET_KEY")
    if app_secret_is_secure(value):
        os.environ["APP_SECRET_KEY"] = str(value)
        return False
    generated = secrets.token_urlsafe(48)
    write({"APP_SECRET_KEY": generated})
    os.environ["APP_SECRET_KEY"] = generated
    return True


def apply_to_process_environment() -> dict[str, str]:
    """Refresh allowed environment variables from the project .env file.

    Local mode treats the project .env as authoritative so an Environment-page
    edit is not shadowed by stale PG_* variables inherited from PowerShell.
    """
    values = _parse()
    for key, value in values.items():
        if key in ALLOWED:
            os.environ[key] = value
    return values
