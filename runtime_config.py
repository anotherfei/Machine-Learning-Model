"""Runtime-editable operational settings backed by Postgres with config.py fallbacks."""
from __future__ import annotations

import json
import threading
from typing import Any

import config

_DEFAULTS = {
    "MAINTENANCE_PROB_URGENT": config.MAINTENANCE_PROB_URGENT,
    "MAINTENANCE_PROB_PLAN": config.MAINTENANCE_PROB_PLAN,
    "FAILURE_HEALTH_THRESHOLD": config.FAILURE_HEALTH_THRESHOLD,
    "MAINTENANCE_HEALTH_INSPECT": config.MAINTENANCE_HEALTH_INSPECT,
    "RETRAIN_BATCH_SIZE": 50,
    "RETRAIN_TIME_CAP_DAYS": 30,
    "REFERENCE_WINDOW_MONTHS": 6,
    "REFERENCE_DEDUP_WINDOW_HOURS": 24,
    "REFERENCE_COSINE_SIMILARITY": 0.98,
    "NEAR_MISS_REGRESSION_WINDOW_HOURS": 1,
    "AUTO_RETRAIN_ENABLED": True,
}

_lock = threading.RLock()
_cache = dict(_DEFAULTS)


def defaults() -> dict[str, Any]:
    return dict(_DEFAULTS)


def get(key: str, default: Any = None) -> Any:
    with _lock:
        return _cache.get(key, _DEFAULTS.get(key, default))


def set_local(values: dict[str, Any]) -> None:
    """Update the process-local cache; used after DB reads and in tests."""
    with _lock:
        _cache.update(values)


def validate_thresholds(values: dict[str, Any]) -> dict[str, float]:
    merged = {k: values.get(k, get(k)) for k in (
        "MAINTENANCE_PROB_URGENT", "MAINTENANCE_PROB_PLAN",
        "FAILURE_HEALTH_THRESHOLD", "MAINTENANCE_HEALTH_INSPECT",
    )}
    merged = {k: float(v) for k, v in merged.items()}
    if not (0 <= merged["MAINTENANCE_PROB_PLAN"] < merged["MAINTENANCE_PROB_URGENT"] <= 1):
        raise ValueError("Require 0 <= MAINTENANCE_PROB_PLAN < MAINTENANCE_PROB_URGENT <= 1")
    if not (0 <= merged["FAILURE_HEALTH_THRESHOLD"] < merged["MAINTENANCE_HEALTH_INSPECT"] <= 100):
        raise ValueError("Require 0 <= FAILURE_HEALTH_THRESHOLD < MAINTENANCE_HEALTH_INSPECT <= 100")
    return merged


TRAINING_KEYS = (
    "RETRAIN_BATCH_SIZE", "RETRAIN_TIME_CAP_DAYS", "REFERENCE_WINDOW_MONTHS",
    "REFERENCE_DEDUP_WINDOW_HOURS", "REFERENCE_COSINE_SIMILARITY",
)


def validate_training_config(values: dict[str, Any]) -> dict[str, float]:
    """
    These already drove retrain_service.run_shadow_retrain() before this
    validator existed (see should_retrain()/_dedup()/_current_reference_features())
    — they just weren't editable from anywhere but a direct DB write.
    This only validates the values are sane; it doesn't change what they do.
    """
    merged = {k: values.get(k, get(k)) for k in TRAINING_KEYS}
    merged = {k: float(v) for k, v in merged.items()}
    if merged["RETRAIN_BATCH_SIZE"] < 1:
        raise ValueError("RETRAIN_BATCH_SIZE must be >= 1")
    if merged["RETRAIN_TIME_CAP_DAYS"] < 1:
        raise ValueError("RETRAIN_TIME_CAP_DAYS must be >= 1")
    if merged["REFERENCE_WINDOW_MONTHS"] < 1:
        raise ValueError("REFERENCE_WINDOW_MONTHS must be >= 1")
    if merged["REFERENCE_DEDUP_WINDOW_HOURS"] < 0:
        raise ValueError("REFERENCE_DEDUP_WINDOW_HOURS must be >= 0")
    if not (0 <= merged["REFERENCE_COSINE_SIMILARITY"] <= 1):
        raise ValueError("REFERENCE_COSINE_SIMILARITY must be in [0, 1]")
    return merged


def load_from_db(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT key, value FROM runtime_config")
        values = {}
        for key, value in cur.fetchall():
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            values[key] = value
    set_local(values)
    return values


def save_to_db(conn, values: dict[str, Any], updated_by: str | None = None) -> None:
    if any(k in values for k in ("MAINTENANCE_PROB_URGENT", "MAINTENANCE_PROB_PLAN", "FAILURE_HEALTH_THRESHOLD", "MAINTENANCE_HEALTH_INSPECT")):
        validate_thresholds(values)
    if any(k in values for k in TRAINING_KEYS):
        validate_training_config(values)
    with conn.cursor() as cur:
        for key, value in values.items():
            cur.execute(
                """INSERT INTO runtime_config(key,value,updated_at,updated_by)
                   VALUES (%s,%s::jsonb,now(),%s)
                   ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value, updated_at=now(), updated_by=EXCLUDED.updated_by""",
                (key, json.dumps(value), updated_by),
            )
        cur.execute("NOTIFY config_changed")
    conn.commit()
    set_local(values)
