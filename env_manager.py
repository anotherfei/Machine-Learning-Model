"""Atomic, masked .env read/write with an allow-list."""
from __future__ import annotations
import os
from pathlib import Path

ENV_PATH = Path(__file__).resolve().parent / ".env"
MASK = "••••••••"
REQUIRED = ("PG_HOST", "PG_DATABASE", "PG_USER", "PG_PASSWORD", "PG_TABLE")
ALLOWED = set(REQUIRED + ("PG_PORT", "PG_COL_TIMESTAMP", "PG_COL_MACHINE_ID", "DEFAULT_MACHINE_ID", "PG_COL_A_RMS_MPS2", "PG_COL_V_RMS_MMS", "PG_COL_A_PEAK_MPS2", "PG_COL_CREST_FACTOR", "PG_COL_TEMPERATURE_C", "APP_SECRET_KEY", "FRONTEND_ORIGIN", "COOKIE_SECURE", "BOOTSTRAP_ADMIN_USER", "BOOTSTRAP_ADMIN_PASSWORD"))
SECRET_KEYS = {"PG_PASSWORD"}


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
