"""Runtime-editable operational settings backed by Postgres with config.py fallbacks."""
from __future__ import annotations

import json
import threading
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

import config

_DEFAULTS = {
    "MAINTENANCE_PROB_URGENT": config.MAINTENANCE_PROB_URGENT,
    "MAINTENANCE_PROB_PLAN": config.MAINTENANCE_PROB_PLAN,
    "FAILURE_HEALTH_THRESHOLD": config.FAILURE_HEALTH_THRESHOLD,
    "MAINTENANCE_HEALTH_INSPECT": config.MAINTENANCE_HEALTH_INSPECT,
    "MAINTENANCE_HORIZON_DAYS": config.MAINTENANCE_HORIZON_DAYS,
    "MAINTENANCE_URGENT_HORIZON_DAYS": config.MAINTENANCE_URGENT_HORIZON_DAYS,
    "TREND_MIN_POINTS": config.TREND_MIN_POINTS,
    "TREND_SLOPE_Z_THRESHOLD": config.TREND_SLOPE_Z_THRESHOLD,
    "TREND_SETTLE_TICKS": config.TREND_SETTLE_TICKS,
    "MAINTENANCE_WARN_CONFIRM_MINUTES": config.MAINTENANCE_WARN_CONFIRM_MINUTES,
    "MAINTENANCE_CRITICAL_CONFIRM_MINUTES": config.MAINTENANCE_CRITICAL_CONFIRM_MINUTES,
    "MAINTENANCE_RECOVERY_MINUTES": config.MAINTENANCE_RECOVERY_MINUTES,
    "SOURCE_STALE_SECONDS": config.SOURCE_STALE_SECONDS,
    "OPERATING_STATE_STOP_CONFIRM_TICKS": config.OPERATING_STATE_STOP_CONFIRM_TICKS,
    "OPERATING_STATE_START_CONFIRM_TICKS": config.OPERATING_STATE_START_CONFIRM_TICKS,
    "HEALTH_SENSITIVITY_STD": config.HEALTH_SENSITIVITY_STD,
    "KALMAN_INIT_SAMPLES": config.KALMAN_INIT_SAMPLES,
    "TREND_LOOKBACK_MINUTES": config.TREND_LOOKBACK_MINUTES,
    "WORKER_POLL_SECONDS": config.WORKER_POLL_SECONDS,
    "NEAR_MISS_TREND_WINDOW_HOURS": config.NEAR_MISS_TREND_WINDOW_HOURS,
    "RETRAIN_BATCH_SIZE": config.RETRAIN_BATCH_SIZE,
    "RETRAIN_TIME_CAP_DAYS": config.RETRAIN_TIME_CAP_DAYS,
    "REFERENCE_WINDOW_MONTHS": config.REFERENCE_WINDOW_MONTHS,
    "REFERENCE_DEDUP_WINDOW_HOURS": config.REFERENCE_DEDUP_WINDOW_HOURS,
    "REFERENCE_COSINE_SIMILARITY": config.REFERENCE_COSINE_SIMILARITY,
    "RETRAIN_CHECK_INTERVAL_MINUTES": config.RETRAIN_CHECK_INTERVAL_MINUTES,
    "RETRAIN_RETRY_COOLDOWN_HOURS": config.RETRAIN_RETRY_COOLDOWN_HOURS,
    "RETRAIN_MAX_FP_RATE_INCREASE": config.RETRAIN_MAX_FP_RATE_INCREASE,
    "AUTO_RETRAIN_ENABLED": config.AUTO_RETRAIN_ENABLED,
}

THRESHOLD_KEYS = (
    "FAILURE_HEALTH_THRESHOLD",
    "MAINTENANCE_HEALTH_INSPECT",
    "MAINTENANCE_PROB_PLAN",
    "MAINTENANCE_PROB_URGENT",
    "MAINTENANCE_HORIZON_DAYS",
    "MAINTENANCE_URGENT_HORIZON_DAYS",
    "TREND_MIN_POINTS",
    "TREND_SLOPE_Z_THRESHOLD",
    "TREND_SETTLE_TICKS",
    "MAINTENANCE_WARN_CONFIRM_MINUTES",
    "MAINTENANCE_CRITICAL_CONFIRM_MINUTES",
    "MAINTENANCE_RECOVERY_MINUTES",
    "OPERATING_STATE_STOP_CONFIRM_TICKS",
    "OPERATING_STATE_START_CONFIRM_TICKS",
    "SOURCE_STALE_SECONDS",
    "HEALTH_SENSITIVITY_STD",
    "KALMAN_INIT_SAMPLES",
    "TREND_LOOKBACK_MINUTES",
    "WORKER_POLL_SECONDS",
    "NEAR_MISS_TREND_WINDOW_HOURS",
)

_lock = threading.RLock()
_cache = dict(_DEFAULTS)
_context_values: ContextVar[dict[str, Any] | None] = ContextVar(
    "runtime_config_context_values", default=None
)


def defaults() -> dict[str, Any]:
    return dict(_DEFAULTS)


def get(key: str, default: Any = None) -> Any:
    context_values = _context_values.get()
    if context_values is not None and key in context_values:
        return context_values[key]
    with _lock:
        return _cache.get(key, _DEFAULTS.get(key, default))


@contextmanager
def override(values: dict[str, Any]):
    """Pin policy values for one execution context only.

    Historical replay runs in an API scheduler thread. A ContextVar keeps its
    captured policy from leaking into the live worker, API requests, or a
    concurrent retraining check while still letting the unchanged production
    monitor call runtime_config.get().
    """
    token = _context_values.set(dict(values))
    try:
        yield
    finally:
        _context_values.reset(token)


def set_local(values: dict[str, Any]) -> None:
    """Update the process-local cache; used after DB reads and in tests."""
    with _lock:
        _cache.update(values)


def normalize_loaded_values(values: dict[str, Any]) -> dict[str, Any]:
    """Clamp legacy persisted policy that predates the planning/urgent split."""
    normalized = dict(values)
    plan = float(normalized.get("MAINTENANCE_HORIZON_DAYS", get("MAINTENANCE_HORIZON_DAYS")))
    urgent = float(normalized.get(
        "MAINTENANCE_URGENT_HORIZON_DAYS", get("MAINTENANCE_URGENT_HORIZON_DAYS")
    ))
    allowed_urgent = sorted(
        float(value) for value in config.FAILURE_PROB_HORIZONS_DAYS
        if float(value) <= config.MAINTENANCE_URGENT_HORIZON_MAX_DAYS
        and float(value) < plan
    )
    if urgent not in allowed_urgent:
        if not allowed_urgent:
            raise ValueError(
                "Planned-maintenance horizon must leave a shorter supported urgent horizon"
            )
        normalized["MAINTENANCE_URGENT_HORIZON_DAYS"] = allowed_urgent[-1]
    return normalized


def validate_thresholds(values: dict[str, Any]) -> dict[str, float | int]:
    merged = {k: values.get(k, get(k)) for k in THRESHOLD_KEYS}
    merged = {k: float(v) for k, v in merged.items()}
    if not (0 <= merged["MAINTENANCE_PROB_PLAN"] < merged["MAINTENANCE_PROB_URGENT"] <= 1):
        raise ValueError("Require 0 <= MAINTENANCE_PROB_PLAN < MAINTENANCE_PROB_URGENT <= 1")
    if not (0 <= merged["FAILURE_HEALTH_THRESHOLD"] < merged["MAINTENANCE_HEALTH_INSPECT"] <= 100):
        raise ValueError("Require 0 <= FAILURE_HEALTH_THRESHOLD < MAINTENANCE_HEALTH_INSPECT <= 100")
    allowed_horizons = {float(value) for value in config.FAILURE_PROB_HORIZONS_DAYS}
    for key in ("MAINTENANCE_HORIZON_DAYS", "MAINTENANCE_URGENT_HORIZON_DAYS"):
        if merged[key] not in allowed_horizons:
            raise ValueError(f"{key} must be one of {sorted(allowed_horizons)} days")
    if merged["MAINTENANCE_URGENT_HORIZON_DAYS"] > config.MAINTENANCE_URGENT_HORIZON_MAX_DAYS:
        raise ValueError("Urgent forecast horizon cannot exceed 1 day")
    if merged["MAINTENANCE_URGENT_HORIZON_DAYS"] >= merged["MAINTENANCE_HORIZON_DAYS"]:
        raise ValueError("Urgent forecast horizon must be shorter than planned-maintenance horizon")
    if merged["TREND_SLOPE_Z_THRESHOLD"] <= 0:
        raise ValueError("TREND_SLOPE_Z_THRESHOLD must be greater than 0")
    if not 0.5 <= merged["HEALTH_SENSITIVITY_STD"] <= 20:
        raise ValueError("HEALTH_SENSITIVITY_STD must be between 0.5 and 20")
    if not 0.25 <= merged["NEAR_MISS_TREND_WINDOW_HOURS"] <= 168:
        raise ValueError("NEAR_MISS_TREND_WINDOW_HOURS must be between 0.25 and 168")
    integer_minimums = {
        "TREND_MIN_POINTS": 3,
        "TREND_SETTLE_TICKS": 0,
        "OPERATING_STATE_STOP_CONFIRM_TICKS": 1,
        "OPERATING_STATE_START_CONFIRM_TICKS": 1,
        "SOURCE_STALE_SECONDS": 1,
        "KALMAN_INIT_SAMPLES": 1,
        "TREND_LOOKBACK_MINUTES": 5,
        "WORKER_POLL_SECONDS": 1,
    }
    for key, minimum in integer_minimums.items():
        if not merged[key].is_integer() or merged[key] < minimum:
            raise ValueError(f"{key} must be a whole number >= {minimum}")
        merged[key] = int(merged[key])
    if merged["KALMAN_INIT_SAMPLES"] > 10000:
        raise ValueError("KALMAN_INIT_SAMPLES cannot exceed 10000 readings")
    if merged["TREND_LOOKBACK_MINUTES"] > 60 * 24 * 30:
        raise ValueError("TREND_LOOKBACK_MINUTES cannot exceed 30 days")
    if merged["WORKER_POLL_SECONDS"] > 3600:
        raise ValueError("WORKER_POLL_SECONDS cannot exceed 3600 seconds")
    timing_keys = (
        "MAINTENANCE_CRITICAL_CONFIRM_MINUTES",
        "MAINTENANCE_WARN_CONFIRM_MINUTES",
        "MAINTENANCE_RECOVERY_MINUTES",
    )
    if any(not 0 <= merged[key] <= 1440 for key in timing_keys):
        raise ValueError("Maintenance confirmation and recovery durations must be between 0 and 1440 minutes")
    if not (
        merged["MAINTENANCE_WARN_CONFIRM_MINUTES"]
        <= merged["MAINTENANCE_CRITICAL_CONFIRM_MINUTES"]
        <= merged["MAINTENANCE_RECOVERY_MINUTES"]
    ):
        raise ValueError(
            "Warning confirmation must not exceed critical confirmation, and "
            "critical confirmation must not exceed recovery duration"
        )
    return merged


TRAINING_KEYS = (
    "RETRAIN_BATCH_SIZE", "RETRAIN_TIME_CAP_DAYS", "REFERENCE_WINDOW_MONTHS",
    "REFERENCE_DEDUP_WINDOW_HOURS", "REFERENCE_COSINE_SIMILARITY",
    "RETRAIN_CHECK_INTERVAL_MINUTES",
    "RETRAIN_RETRY_COOLDOWN_HOURS", "RETRAIN_MAX_FP_RATE_INCREASE",
)


def validate_training_config(values: dict[str, Any]) -> dict[str, float | int]:
    """
    These already drove retrain_service.run_shadow_retrain() before this
    validator existed (see should_retrain()/_dedup_within_machine()/reference balancing)
    — they just weren't editable from anywhere but a direct DB write.
    This only validates the values are sane; it doesn't change what they do.
    """
    merged = {k: values.get(k, get(k)) for k in TRAINING_KEYS}
    merged = {k: float(v) for k, v in merged.items()}
    integer_limits = {
        "RETRAIN_BATCH_SIZE": (1, 100000),
        "RETRAIN_TIME_CAP_DAYS": (1, 3650),
        "REFERENCE_WINDOW_MONTHS": (1, 120),
        "RETRAIN_CHECK_INTERVAL_MINUTES": (5, 1440),
    }
    for key, (minimum, maximum) in integer_limits.items():
        if not merged[key].is_integer() or not minimum <= merged[key] <= maximum:
            raise ValueError(f"{key} must be a whole number between {minimum} and {maximum}")
        merged[key] = int(merged[key])
    if merged["RETRAIN_TIME_CAP_DAYS"] > merged["REFERENCE_WINDOW_MONTHS"] * 28:
        raise ValueError(
            "RETRAIN_TIME_CAP_DAYS must fit within REFERENCE_WINDOW_MONTHS so candidates do not expire first"
        )
    if not 0 <= merged["REFERENCE_DEDUP_WINDOW_HOURS"] <= 8760:
        raise ValueError("REFERENCE_DEDUP_WINDOW_HOURS must be between 0 and 8760")
    if not (0 <= merged["REFERENCE_COSINE_SIMILARITY"] <= 1):
        raise ValueError("REFERENCE_COSINE_SIMILARITY must be in [0, 1]")
    if not 0.25 <= merged["RETRAIN_RETRY_COOLDOWN_HOURS"] <= 720:
        raise ValueError("RETRAIN_RETRY_COOLDOWN_HOURS must be between 0.25 and 720")
    if not 0 <= merged["RETRAIN_MAX_FP_RATE_INCREASE"] <= 0.10:
        raise ValueError("RETRAIN_MAX_FP_RATE_INCREASE must be between 0 and 0.10")
    return merged


def load_from_db(conn) -> dict[str, Any]:
    with conn.cursor() as cur:
        cur.execute("SELECT key, value FROM state WHERE namespace='config'")
        values = {}
        for key, value in cur.fetchall():
            if isinstance(value, str):
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    pass
            values[key] = value
    values = normalize_loaded_values(values)
    set_local(values)
    return values


def save_to_db(conn, values: dict[str, Any], updated_by: str | None = None) -> None:
    if any(k in values for k in THRESHOLD_KEYS):
        validate_thresholds(values)
    if any(k in values for k in TRAINING_KEYS):
        validate_training_config(values)
    with conn.cursor() as cur:
        for key, value in values.items():
            cur.execute(
                """INSERT INTO state(namespace,key,value,updated_at,updated_by)
                   VALUES ('config',%s,%s::jsonb,now(),%s)
                   ON CONFLICT(namespace,key) DO UPDATE SET
                     value=EXCLUDED.value,updated_at=now(),updated_by=EXCLUDED.updated_by""",
                (key, json.dumps(value), updated_by),
            )
        cur.execute("NOTIFY config_changed")
    conn.commit()
    set_local(values)
