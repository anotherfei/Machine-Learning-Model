"""Temporary local demo API backed by SQLite.

Run via `start_project.ps1 -Mock`. This module intentionally bypasses the
production PostgreSQL/ML worker so the web interface can be verified on a
machine that has no database server or trained artifacts yet.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import random
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
import artifact_utils
import env_manager
import runtime_config
from api.auth import User, current_user, hash_password, issue_cookie, require_admin, session_user, verify_password, COOKIE

# Mock mode is bound to localhost, but it still receives a fresh non-placeholder
# signing key for each process instead of using the production fallback.
if not env_manager.app_secret_is_secure(os.getenv("APP_SECRET_KEY")):
    os.environ["APP_SECRET_KEY"] = secrets.token_urlsafe(48)

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = Path(__file__).resolve().parent / "mock_demo.db"
_LOCK = threading.RLock()
_RNG = random.Random(42)
_login_attempts = defaultdict(deque)
MACHINE_IDS = ("MACHINE-001", "MACHINE-002", "MACHINE-003")
LEGACY_MACHINE_IDS = dict(zip(("VVB001", "VVB002", "VVB003"), MACHINE_IDS))
_live_index: dict[str, int] = {}

app = FastAPI(title="Spindle Condition Monitoring API (Mock Demo)", version="1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _connect() -> sqlite3.Connection:
    c = sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _maintenance_for(health: float, forecast_risk: float) -> tuple[str, str, str]:
    if health <= runtime_config.get("FAILURE_HEALTH_THRESHOLD", 20):
        return "CRITICAL", "Health is below the failure threshold.", "health_threshold"
    if health <= runtime_config.get("MAINTENANCE_HEALTH_INSPECT", 30):
        return "WARN", "Health is below the inspection threshold.", "health_inspect"
    if forecast_risk >= runtime_config.get("MAINTENANCE_PROB_URGENT", 0.80):
        return "CRITICAL", "Mock forecast boundary-crossing risk is above the urgent threshold.", "trend_probability"
    if forecast_risk >= runtime_config.get("MAINTENANCE_PROB_PLAN", 0.60):
        return "WARN", "Mock forecast boundary-crossing risk indicates maintenance should be planned.", "trend_probability"
    return "OK", "No maintenance action is currently required.", "none"


def _sensor_values(i: int, machine_id: str) -> dict[str, float]:
    # Slowly wandering, deterministic demo signal with occasional stronger vibration.
    unit_offset = MACHINE_IDS.index(machine_id) if machine_id in MACHINE_IDS else 0
    phase = i / 11.0 + unit_offset * 0.7
    bump = 1.3 if i % 37 in (0, 1, 2) else 0.0
    return {
        "a_rms_mps2": round(1.15 + 0.28 * math.sin(phase) + bump, 4),
        "v_rms_mms": round(2.25 + 0.45 * math.sin(phase / 1.4) + bump * 0.7, 4),
        "a_peak_mps2": round(3.8 + 0.9 * math.sin(phase / 0.8) + bump * 2.2, 4),
        "crest_factor": round(3.0 + 0.25 * math.cos(phase) + bump * 0.18, 4),
        "temperature_c": round(37.5 + 2.1 * math.sin(phase / 2.0) + bump * 1.8, 3),
    }


def _prediction(i: int, machine_id: str = "MACHINE-001", when: datetime | None = None) -> dict:
    when = when or _now()
    raw = _sensor_values(i, machine_id)
    # Keep demo values visually useful: mostly healthy, with periodic warnings/critical points.
    cycle = i % 70
    if cycle in (55, 56):
        health = 17.0 + (cycle - 55) * 2.0
    elif 48 <= cycle <= 54:
        health = 27.0 + (54 - cycle) * 1.2
    else:
        health = 78.0 + 10.0 * math.sin(i / 18.0)
    health = max(5.0, min(99.0, health))
    anomaly = max(0.01, min(0.99, (100.0 - health) / 100.0 + 0.08 * abs(math.sin(i / 7.0))))
    probability = {
        horizon: round(min(0.99, anomaly * (0.25 + 0.55 * float(horizon))), 4)
        for horizon in config.FAILURE_PROB_HORIZONS_DAYS
    }
    horizon = float(runtime_config.get("MAINTENANCE_HORIZON_DAYS", config.MAINTENANCE_HORIZON_DAYS))
    selected_horizon = min(probability, key=lambda value: abs(float(value) - horizon))
    level, reason, trigger = _maintenance_for(health, probability[selected_horizon])
    return {
        "machine_id": machine_id,
        "tick_timestamp": _iso(when),
        "model_version": "mock-demo-v1",
        "raw_reading": raw,
        "anomaly_score": round(anomaly, 4),
        "health_raw": round(health + _RNG.uniform(-2.0, 2.0), 3),
        "health_state": round(health, 3),
        "trend_slope_per_day": round(-0.08 * math.sin(i / 15.0), 5),
        "remaining_days": round(max(1.0, health / 3.5), 2),
        "failure_probability": probability,
        "maintenance_level": level,
        "maintenance_reason": reason,
        "maintenance_trigger": trigger,
        "top_contributors": ["a_rms_mps2", "v_rms_mms"] if level != "OK" else ["temperature_c"],
    }


def _insert_prediction(c: sqlite3.Connection, row: dict, create_alert: bool = False) -> int:
    raw_json = json.dumps(row["raw_reading"])
    c.execute(
        """INSERT INTO spindle_predictions(
        machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,
        trend_slope_per_day,remaining_days,failure_probability,maintenance_level,
        maintenance_reason,maintenance_trigger,top_contributors,is_backfill)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
        (
            row["machine_id"], row["tick_timestamp"], row["model_version"], raw_json, row["anomaly_score"], row["health_raw"],
            row["health_state"], row["trend_slope_per_day"], row["remaining_days"],
            json.dumps(row["failure_probability"]), row["maintenance_level"], row["maintenance_reason"],
            row["maintenance_trigger"], json.dumps(row["top_contributors"]),
        ),
    )
    pid = int(c.execute("SELECT last_insert_rowid()").fetchone()[0])
    if create_alert and row["maintenance_level"] in ("WARN", "CRITICAL"):
        c.execute(
            """INSERT INTO alerts(machine_id,tick_timestamp,model_version,trigger,level,health_state,anomaly_score,raw_reading,status)
               VALUES(?,?,?,?,?,?,?,?,'pending')""",
            (row["machine_id"], row["tick_timestamp"], row["model_version"], row["maintenance_trigger"], row["maintenance_level"],
             row["health_state"], row["anomaly_score"], raw_json),
        )
    return pid


def initialize_mock_database(reset: bool = False) -> Path:
    global _live_index
    with _LOCK:
        if reset and DB_PATH.exists():
            DB_PATH.unlink()
        c = _connect()
        c.executescript(
            """
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS app_users(
              username TEXT PRIMARY KEY, password_hash TEXT NOT NULL, role TEXT NOT NULL, disabled INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS runtime_config(key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_by TEXT);
            CREATE TABLE IF NOT EXISTS alerts(
              id INTEGER PRIMARY KEY AUTOINCREMENT, machine_id TEXT NOT NULL DEFAULT 'MACHINE-001', tick_timestamp TEXT NOT NULL, model_version TEXT NOT NULL,
              trigger TEXT NOT NULL, level TEXT NOT NULL, health_state REAL NOT NULL, anomaly_score REAL NOT NULL,
              raw_reading TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', reviewed_by TEXT, reviewed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS reference_candidates(
              id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER UNIQUE, tick_timestamp TEXT NOT NULL,
              added_to_reference_at TEXT
            );
            CREATE TABLE IF NOT EXISTS model_versions(
              version_id TEXT PRIMARY KEY, artifact_path TEXT NOT NULL, reference_signature TEXT NOT NULL,
              status TEXT NOT NULL, validation_report TEXT, created_at TEXT NOT NULL, promoted_at TEXT, promoted_by TEXT
            );
            CREATE TABLE IF NOT EXISTS model_calibrations(
              id INTEGER PRIMARY KEY AUTOINCREMENT, version_id TEXT NOT NULL, machine_id TEXT NOT NULL DEFAULT 'MACHINE-001', calibration TEXT NOT NULL,
              source_rows INTEGER NOT NULL, source_description TEXT, created_at TEXT NOT NULL, created_by TEXT
            );
            CREATE TABLE IF NOT EXISTS machine_model_calibrations(
              machine_id TEXT NOT NULL, version_id TEXT NOT NULL, calibration_id INTEGER NOT NULL,
              updated_at TEXT NOT NULL, PRIMARY KEY(machine_id,version_id)
            );
            CREATE TABLE IF NOT EXISTS regression_tests(
              id INTEGER PRIMARY KEY AUTOINCREMENT, machine_id TEXT NOT NULL DEFAULT 'MACHINE-001', description TEXT NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
              minimum_anomaly_risk REAL NOT NULL, source_alert_id INTEGER,
              target_prediction_id INTEGER,target_timestamp TEXT,
              created_at TEXT NOT NULL, created_by TEXT, disabled_at TEXT, disabled_by TEXT
            );
            CREATE TABLE IF NOT EXISTS model_version_candidates(
              version_id TEXT NOT NULL, candidate_id INTEGER NOT NULL,
              created_at TEXT NOT NULL DEFAULT (datetime('now')), PRIMARY KEY(version_id,candidate_id)
            );
            CREATE TABLE IF NOT EXISTS retrain_jobs(
              id INTEGER PRIMARY KEY AUTOINCREMENT, trigger TEXT NOT NULL, requested_by TEXT,
              force INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL, candidate_signature TEXT,
              model_version_id TEXT, result TEXT, error TEXT, created_at TEXT NOT NULL,
              started_at TEXT, finished_at TEXT
            );
            CREATE TABLE IF NOT EXISTS mock_env(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS spindle_predictions(
              id INTEGER PRIMARY KEY AUTOINCREMENT, machine_id TEXT NOT NULL DEFAULT 'MACHINE-001', tick_timestamp TEXT NOT NULL, model_version TEXT NOT NULL,
              raw_reading TEXT NOT NULL, anomaly_score REAL NOT NULL, health_raw REAL NOT NULL, health_state REAL NOT NULL,
              trend_slope_per_day REAL NOT NULL, remaining_days REAL NOT NULL, failure_probability TEXT NOT NULL,
              maintenance_level TEXT NOT NULL, maintenance_reason TEXT NOT NULL, maintenance_trigger TEXT NOT NULL,
              top_contributors TEXT, is_backfill INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE IF NOT EXISTS near_miss_reviews(
              id INTEGER PRIMARY KEY AUTOINCREMENT, prediction_id INTEGER UNIQUE NOT NULL,
              status TEXT NOT NULL DEFAULT 'pending', reviewed_by TEXT, reviewed_at TEXT,
              created_at TEXT NOT NULL DEFAULT (datetime('now'))
            );
            """
        )
        # SQLite has no "ADD COLUMN IF NOT EXISTS"; guard manually so this
        # (idempotent, run on every startup) doesn't fail with "duplicate
        # column name" on the second run. NULL = "normal"/pooled
        # calibration — same convention as model_registry.py's Postgres
        # column of the same name.
        existing_cols = {r[1] for r in c.execute("PRAGMA table_info(model_versions)").fetchall()}
        if "active_calibration_id" not in existing_cols:
            c.execute("ALTER TABLE model_versions ADD COLUMN active_calibration_id INTEGER")
        calibration_cols = {r[1] for r in c.execute("PRAGMA table_info(model_calibrations)").fetchall()}
        if "machine_id" not in calibration_cols:
            c.execute("ALTER TABLE model_calibrations ADD COLUMN machine_id TEXT NOT NULL DEFAULT 'MACHINE-001'")
        candidate_cols = {r[1] for r in c.execute("PRAGMA table_info(reference_candidates)").fetchall()}
        if "added_to_reference_at" not in candidate_cols:
            c.execute("ALTER TABLE reference_candidates ADD COLUMN added_to_reference_at TEXT")
        regression_cols = {r[1] for r in c.execute("PRAGMA table_info(regression_tests)").fetchall()}
        if "disabled_at" not in regression_cols:
            c.execute("ALTER TABLE regression_tests ADD COLUMN disabled_at TEXT")
        if "disabled_by" not in regression_cols:
            c.execute("ALTER TABLE regression_tests ADD COLUMN disabled_by TEXT")
        if "source_alert_id" not in regression_cols:
            c.execute("ALTER TABLE regression_tests ADD COLUMN source_alert_id INTEGER")
        if "created_by" not in regression_cols:
            c.execute("ALTER TABLE regression_tests ADD COLUMN created_by TEXT")
        if "target_prediction_id" not in regression_cols:
            c.execute("ALTER TABLE regression_tests ADD COLUMN target_prediction_id INTEGER")
        if "target_timestamp" not in regression_cols:
            c.execute("ALTER TABLE regression_tests ADD COLUMN target_timestamp TEXT")
        for table_name in ("alerts", "regression_tests", "spindle_predictions"):
            table_cols = {r[1] for r in c.execute(f"PRAGMA table_info({table_name})").fetchall()}
            if "machine_id" not in table_cols:
                c.execute(f"ALTER TABLE {table_name} ADD COLUMN machine_id TEXT NOT NULL DEFAULT 'MACHINE-001'")
        # Earlier mock builds mistakenly used the VVB001 sensor model as an
        # asset ID. Relabel those demo rows while keeping VVB001 as sensor
        # specification metadata in config.py/preprocessing.py.
        for old_machine_id, new_machine_id in LEGACY_MACHINE_IDS.items():
            for table_name in ("alerts", "regression_tests", "spindle_predictions", "model_calibrations"):
                c.execute(f"UPDATE {table_name} SET machine_id=? WHERE machine_id=?", (new_machine_id, old_machine_id))
            c.execute("""INSERT OR IGNORE INTO machine_model_calibrations(machine_id,version_id,calibration_id,updated_at)
                         SELECT ?,version_id,calibration_id,updated_at FROM machine_model_calibrations WHERE machine_id=?""",
                      (new_machine_id, old_machine_id))
            c.execute("DELETE FROM machine_model_calibrations WHERE machine_id=?", (old_machine_id,))
        c.execute("UPDATE mock_env SET value=? WHERE key='DEFAULT_MACHINE_ID' AND value IN ('VVB001','VVB002','VVB003')", (MACHINE_IDS[0],))
        c.execute("""INSERT OR IGNORE INTO machine_model_calibrations(machine_id,version_id,calibration_id,updated_at)
                     SELECT 'MACHINE-001',version_id,active_calibration_id,datetime('now') FROM model_versions
                     WHERE active_calibration_id IS NOT NULL""")
        c.execute("UPDATE model_versions SET active_calibration_id=NULL WHERE active_calibration_id IS NOT NULL")
        if c.execute("SELECT count(*) FROM app_users").fetchone()[0] == 0:
            username = os.getenv("BOOTSTRAP_ADMIN_USER", "admin")
            password = os.getenv("BOOTSTRAP_ADMIN_PASSWORD", "change-me-on-first-deployment")
            c.execute("INSERT INTO app_users(username,password_hash,role) VALUES(?,?,?)", (username, hash_password(password), "admin"))
        for key, value in runtime_config.defaults().items():
            c.execute("INSERT OR IGNORE INTO runtime_config(key,value) VALUES(?,?)", (key, json.dumps(value)))
        if c.execute("SELECT count(*) FROM mock_env").fetchone()[0] == 0:
            vals = {
                "PG_HOST": "localhost", "PG_PORT": "5432", "PG_DATABASE": "mock-demo",
                "PG_USER": "demo", "PG_PASSWORD": "not-used-in-mock-mode", "PG_TABLE": "mock_sensor_readings",
                "PG_COL_MACHINE_ID": "machine_id", "DEFAULT_MACHINE_ID": MACHINE_IDS[0],
                "FRONTEND_ORIGIN": "http://localhost:5173",
                "APP_SESSION_SECONDS": "28800",
            }
            c.executemany("INSERT INTO mock_env(key,value) VALUES(?,?)", vals.items())
        if c.execute("SELECT count(*) FROM model_versions").fetchone()[0] == 0:
            now = _iso(_now())
            c.execute(
                """INSERT INTO model_versions
                   (version_id,artifact_path,reference_signature,status,validation_report,created_at,promoted_at,promoted_by)
                   VALUES(?,?,?,?,?,?,?,?)""",
                ("mock-demo-v1", "mock://artifacts/mock-demo-v1", "mock-reference", "active", json.dumps({"mode":"mock","passed":True}), now, now, "bootstrap"),
            )
            c.execute(
                """INSERT INTO model_versions
                   (version_id,artifact_path,reference_signature,status,validation_report,created_at,promoted_at,promoted_by)
                   VALUES(?,?,?,?,?,?,?,?)""",
                ("mock-shadow-v2", "mock://artifacts/mock-shadow-v2", "mock-reference-v2", "shadow", json.dumps({"mode":"mock","passed":True,"note":"demo shadow model"}), now, None, None),
            )
        start = _now() - timedelta(minutes=119)
        for machine_id in MACHINE_IDS:
            count = int(c.execute("SELECT count(*) FROM spindle_predictions WHERE machine_id=?", (machine_id,)).fetchone()[0])
            if count == 0:
                for i in range(120):
                    row = _prediction(i, machine_id, start + timedelta(minutes=i))
                    _insert_prediction(c, row, create_alert=(row["maintenance_level"] != "OK" and i % 3 == 0))
                count = 120
            _live_index[machine_id] = count
        # Older Demo builds used sklearn's negative raw anomaly score as a
        # probability and clamped almost every auto-created floor to 0.05.
        # Repair only those recognizable generated rows.
        for test in c.execute("SELECT id,description FROM regression_tests WHERE minimum_anomaly_risk=0.05 AND description LIKE 'Near-miss%prediction #%'").fetchall():
            match = re.search(r"prediction #(\d+)$", test["description"])
            if not match:
                continue
            prediction = c.execute("SELECT health_state FROM spindle_predictions WHERE id=?", (int(match.group(1)),)).fetchone()
            if prediction:
                floor = runtime_config.regression_risk_floor(prediction["health_state"])
                c.execute("UPDATE regression_tests SET minimum_anomaly_risk=? WHERE id=?", (floor, test["id"]))
        for test in c.execute(
            """SELECT id,description FROM regression_tests
               WHERE target_prediction_id IS NULL AND description LIKE 'Near-miss%prediction #%'
            """
        ).fetchall():
            match = re.search(r"prediction #(\d+)$", test["description"])
            if not match:
                continue
            prediction_id = int(match.group(1))
            prediction = c.execute(
                "SELECT tick_timestamp FROM spindle_predictions WHERE id=?", (prediction_id,)
            ).fetchone()
            if prediction:
                c.execute(
                    """UPDATE regression_tests SET target_prediction_id=?,target_timestamp=?
                       WHERE id=?""",
                    (prediction_id, prediction["tick_timestamp"], test["id"]),
                )
        c.commit()
        c.close()
        _load_threshold_cache()
        return DB_PATH


def _load_threshold_cache() -> None:
    c = _connect()
    rows = c.execute("SELECT key,value FROM runtime_config").fetchall()
    c.close()
    vals = {}
    for r in rows:
        try: vals[r["key"]] = json.loads(r["value"])
        except Exception: vals[r["key"]] = r["value"]
    runtime_config.set_local(vals)


def _row_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for key in ("raw_reading", "failure_probability", "top_contributors", "validation_report", "result"):
        if key in d and isinstance(d[key], str):
            try: d[key] = json.loads(d[key])
            except Exception: pass
    return d


@app.on_event("startup")
def startup() -> None:
    initialize_mock_database(reset=False)
    print(f"[mock] SQLite demo database ready: {DB_PATH}")


@app.get("/api/health")
def health():
    return {"ok": True, "mode": "mock"}


class LoginBody(BaseModel): username: str; password: str
class ReviewBody(BaseModel): decision: str
class ThresholdBody(BaseModel):
    MAINTENANCE_PROB_URGENT: float
    MAINTENANCE_PROB_PLAN: float
    FAILURE_HEALTH_THRESHOLD: float
    MAINTENANCE_HEALTH_INSPECT: float
    MAINTENANCE_HORIZON_DAYS: float
    TREND_MIN_POINTS: int
    TREND_SLOPE_Z_THRESHOLD: float
    TREND_SETTLE_TICKS: int
    MAINTENANCE_TREND_DEBOUNCE_TICKS: int
    MAINTENANCE_TREND_RECOVERY_TICKS: int
    OPERATING_STATE_STOP_CONFIRM_TICKS: int
    OPERATING_STATE_START_CONFIRM_TICKS: int
    SOURCE_STALE_SECONDS: int
    HEALTH_SENSITIVITY_STD: float
    KALMAN_INIT_SAMPLES: int
    TREND_LOOKBACK_MINUTES: int
    WORKER_POLL_SECONDS: int
    NEAR_MISS_TREND_WINDOW_HOURS: float
class EnvBody(BaseModel): values: dict[str, str]
class UserCreate(BaseModel): username: str; password: str; role: str = "viewer"
class RegressionBody(BaseModel): description: str; start: str; end: str; machine_id: str = "MACHINE-001"; minimum_anomaly_risk: float = 0.6
class BackfillBody(BaseModel): start: str; end: str; mode: str = "repredict"; machine_id: str | None = None
class TrainingConfigBody(BaseModel):
    RETRAIN_BATCH_SIZE: int
    RETRAIN_TIME_CAP_DAYS: int
    REFERENCE_WINDOW_MONTHS: int
    REFERENCE_DEDUP_WINDOW_HOURS: float
    REFERENCE_COSINE_SIMILARITY: float
    NEAR_MISS_REGRESSION_WINDOW_HOURS: float
    RETRAIN_CHECK_INTERVAL_MINUTES: int
    RETRAIN_RETRY_COOLDOWN_HOURS: float
    RETRAIN_MAX_FP_RATE_INCREASE: float
    AUTO_RETRAIN_ENABLED: bool = True


@app.post("/api/login")
def login(body: LoginBody, response: Response):
    now = time.time(); q = _login_attempts[body.username]
    while q and now - q[0] > 60: q.popleft()
    if len(q) >= 10: raise HTTPException(429, "Too many login attempts; retry shortly")
    q.append(now)
    c = _connect(); row = c.execute("SELECT password_hash,role,disabled FROM app_users WHERE username=?", (body.username,)).fetchone(); c.close()
    if not row or row["disabled"] or not verify_password(body.password, row["password_hash"]):
        raise HTTPException(401, "Invalid credentials")
    q.clear(); issue_cookie(response, body.username, row["role"])
    return {"username": body.username, "role": row["role"], "mock_mode": True}


@app.post("/api/logout")
def logout(response: Response):
    response.delete_cookie(COOKIE); return {"ok": True}


@app.get("/api/me")
def me(user: User = Depends(current_user)):
    return {**user.__dict__, "mock_mode": True}


@app.get("/api/machines")
def machines(user: User = Depends(current_user)):
    c = _connect()
    items = [r[0] for r in c.execute(
        "SELECT machine_id FROM (SELECT DISTINCT machine_id FROM spindle_predictions UNION SELECT DISTINCT machine_id FROM alerts) ORDER BY machine_id"
    ).fetchall() if r[0]]
    c.close()
    return {"items": items, "default": MACHINE_IDS[0], "source": "mock", "table": "mock_demo.db"}


@app.get("/api/fleet/condition-trend")
def fleet_condition_trend(days: int = 7, user: User = Depends(current_user)):
    if not 1 <= days <= 31:
        raise HTTPException(400, "days must be between 1 and 31")
    end = _now()
    start = end - timedelta(days=days)
    c = _connect()
    machine_ids = [row[0] for row in c.execute(
        "SELECT DISTINCT machine_id FROM spindle_predictions ORDER BY machine_id"
    ).fetchall() if row[0]]
    rows = c.execute(
        """WITH ranked_tick AS (
             SELECT machine_id,tick_timestamp,health_state,
                    row_number() OVER (
                      PARTITION BY machine_id,tick_timestamp ORDER BY id DESC
                    ) AS rank
             FROM spindle_predictions
             WHERE julianday(tick_timestamp) >= julianday(?)
               AND julianday(tick_timestamp) <= julianday(?)
           )
           SELECT machine_id,
                  strftime('%Y-%m-%dT%H:00:00Z',tick_timestamp) AS bucket,
                  avg(health_state) AS health_state,
                  count(*) AS samples
           FROM ranked_tick
           WHERE rank=1
           GROUP BY machine_id,strftime('%Y-%m-%dT%H:00:00Z',tick_timestamp)
           ORDER BY machine_id,bucket""",
        (_iso(start), _iso(end)),
    ).fetchall()
    c.close()
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["machine_id"], []).append({
            "timestamp": row["bucket"],
            "health_state": float(row["health_state"]),
            "samples": int(row["samples"]),
        })
    return {
        "days": days, "metric": "hourly_average_condition", "start": _iso(start), "end": _iso(end),
        "series": [{"machine_id": machine_id, "points": grouped.get(machine_id, [])} for machine_id in machine_ids],
    }


@app.get("/api/live/latest")
def latest_live(machine_id: str = MACHINE_IDS[0], user: User = Depends(current_user)):
    c = _connect()
    row = c.execute(
        "SELECT * FROM spindle_predictions WHERE machine_id=? ORDER BY tick_timestamp DESC LIMIT 1",
        (machine_id,),
    ).fetchone()
    c.close()
    if not row:
        raise HTTPException(404, f"No mock rows found for machine {machine_id}")
    row = dict(row)
    raw = json.loads(row["raw_reading"]) if isinstance(row["raw_reading"], str) else row["raw_reading"]
    return {
        "machine_id": machine_id,
        "timestamp": row["tick_timestamp"],
        "prediction_timestamp": row["tick_timestamp"],
        "model_version": row["model_version"],
        "anomaly_score": row["anomaly_score"],
        "health_state": row["health_state"],
        "maintenance": {
            "level": row["maintenance_level"],
            "reason": row["maintenance_reason"],
            "trigger": row["maintenance_trigger"],
        },
        "prediction_available": True,
        "prediction_lag_seconds": 0.0,
        "operating_state": "RUNNING",
        "operating_state_reason": "Synthetic vibration is in the running regime.",
        "operating_state_confidence": 1.0,
        "source": "mock",
        "mock_mode": True,
        **raw,
    }


@app.post("/api/users")
def create_user(body: UserCreate, admin: User = Depends(require_admin)):
    if body.role not in ("viewer", "admin"): raise HTTPException(400, "role must be viewer or admin")
    username = body.username.strip()
    if not 3 <= len(username) <= 64: raise HTTPException(400, "username must contain 3 to 64 characters")
    if not env_manager.bootstrap_password_is_secure(body.password): raise HTTPException(400, "password must be a unique password of at least 12 characters")
    c = _connect()
    try:
        c.execute("INSERT INTO app_users(username,password_hash,role) VALUES(?,?,?)", (username, hash_password(body.password), body.role)); c.commit()
    except sqlite3.IntegrityError:
        raise HTTPException(409, "username already exists")
    finally: c.close()
    return {"username": username, "role": body.role}


@app.get("/api/env")
def get_env(admin: User = Depends(require_admin)):
    c = _connect(); rows = c.execute("SELECT key,value FROM mock_env ORDER BY key").fetchall(); c.close()
    out = {r["key"]: r["value"] for r in rows}
    for key in env_manager.SECRET_KEYS:
        if out.get(key): out[key] = env_manager.MASK
    return out


@app.put("/api/env")
def put_env(body: EnvBody, admin: User = Depends(require_admin)):
    c = _connect(); changed = []
    for key, raw_value in body.values.items():
        if key not in env_manager.ALLOWED:
            c.close()
            raise HTTPException(400, f"Unsupported .env key: {key}")
        value = str(raw_value).replace("\n", "").replace("\r", "")
        if key in env_manager.SECRET_KEYS and value == env_manager.MASK: continue
        if key == "APP_SECRET_KEY" and not env_manager.app_secret_is_secure(value):
            c.close()
            raise HTTPException(400, "APP_SECRET_KEY must be a unique secret of at least 32 characters")
        if key == "BOOTSTRAP_ADMIN_PASSWORD" and value and not env_manager.bootstrap_password_is_secure(value):
            c.close()
            raise HTTPException(400, "BOOTSTRAP_ADMIN_PASSWORD must be a unique password of at least 12 characters")
        if key == "APP_SESSION_SECONDS":
            try:
                seconds = int(value)
            except ValueError:
                c.close()
                raise HTTPException(400, "APP_SESSION_SECONDS must be a whole number")
            if not 300 <= seconds <= 604800:
                c.close()
                raise HTTPException(400, "APP_SESSION_SECONDS must be between 300 and 604800")
            value = str(seconds)
        old = c.execute("SELECT value FROM mock_env WHERE key=?", (key,)).fetchone()
        if old is None or old["value"] != str(value):
            c.execute("INSERT INTO mock_env(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
            changed.append(key)
    c.commit(); c.close()
    return {"changed_keys": changed, "worker_restart_requested": False, "mock_mode": True}


@app.get("/api/config/thresholds")
def thresholds(user: User = Depends(current_user)):
    return {k: runtime_config.get(k) for k in runtime_config.THRESHOLD_KEYS}


@app.put("/api/config/thresholds")
def set_thresholds(body: ThresholdBody, admin: User = Depends(require_admin)):
    try:
        values = runtime_config.validate_thresholds(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    c = _connect()
    for key, value in values.items():
        c.execute("INSERT INTO runtime_config(key,value,updated_by) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_by=excluded.updated_by", (key, json.dumps(value), admin.username))
    c.commit(); c.close(); runtime_config.set_local(values)
    return values


@app.get("/api/config/training")
def training_config(user: User = Depends(current_user)):
    keys = (*runtime_config.TRAINING_KEYS, "AUTO_RETRAIN_ENABLED")
    return {k: runtime_config.get(k) for k in keys}


@app.put("/api/config/training")
def set_training_config(body: TrainingConfigBody, admin: User = Depends(require_admin)):
    try:
        values = runtime_config.validate_training_config(body.model_dump())
    except ValueError as exc:
        raise HTTPException(400, str(exc))
    values["AUTO_RETRAIN_ENABLED"] = body.AUTO_RETRAIN_ENABLED
    c = _connect()
    for key, value in values.items():
        c.execute("INSERT INTO runtime_config(key,value,updated_by) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_by=excluded.updated_by", (key, json.dumps(value), admin.username))
    c.commit(); c.close(); runtime_config.set_local(values)
    # Mock mode has no scheduled background retrain job to gate (see
    # api/main.py's _scheduled_retrain) — "Run shadow retrain" is always a
    # manual click here regardless of this setting. It's still stored and
    # returned so the Models page toggle isn't a dead control.
    return values


@app.get("/api/alerts")
def alerts(status: str = "pending", trigger: str | None = None, machine_id: str = "MACHINE-001", limit: int = 50, offset: int = 0, user: User = Depends(current_user)):
    limit = max(1, min(limit, 500)); offset = max(0, offset)
    count_sql = "SELECT count(*) FROM alerts WHERE status=? AND machine_id=?"; count_args: list = [status, machine_id]
    sql = "SELECT * FROM alerts WHERE status=? AND machine_id=?"; args: list = [status, machine_id]
    if trigger:
        count_sql += " AND trigger=?"; count_args.append(trigger)
        sql += " AND trigger=?"; args.append(trigger)
    sql += " ORDER BY tick_timestamp DESC LIMIT ? OFFSET ?"; args.extend([limit, offset])
    c = _connect()
    total = c.execute(count_sql, count_args).fetchone()[0]
    rows = [_row_dict(r) for r in c.execute(sql, args).fetchall()]
    c.close(); return {"items": rows, "total": total, "limit": limit, "offset": offset}


@app.post("/api/alerts/{alert_id}/review")
def review(alert_id: int, body: ReviewBody, user: User = Depends(current_user)):
    if body.decision not in ("confirmed_anomaly", "confirmed_normal"):
        raise HTTPException(400, "decision must be confirmed_anomaly or confirmed_normal")
    c = _connect(); row = c.execute("SELECT tick_timestamp,status FROM alerts WHERE id=?", (alert_id,)).fetchone()
    if not row or row["status"] != "pending": c.close(); raise HTTPException(409, "Alert not found or already reviewed")
    c.execute("UPDATE alerts SET status=?,reviewed_by=?,reviewed_at=? WHERE id=?", (body.decision, user.username, _iso(_now()), alert_id))
    if body.decision == "confirmed_normal":
        c.execute("INSERT OR IGNORE INTO reference_candidates(alert_id,tick_timestamp) VALUES(?,?)", (alert_id, row["tick_timestamp"]))
    c.commit(); c.close(); return {"id": alert_id, "status": body.decision}


@app.get("/api/alerts/{alert_id}/context")
def alert_context(alert_id: int, hours: float = 3, user: User = Depends(current_user)):
    if not 0.25 <= hours <= 168: raise HTTPException(400, "hours must be between 0.25 and 168")
    c = _connect(); a = c.execute("SELECT tick_timestamp,machine_id FROM alerts WHERE id=?", (alert_id,)).fetchone()
    if not a: c.close(); raise HTTPException(404, "Alert not found")
    center = datetime.fromisoformat(a["tick_timestamp"].replace("Z", "+00:00")); lo = _iso(center - timedelta(hours=hours)); hi = _iso(center + timedelta(hours=hours))
    rows = [dict(r) for r in c.execute("SELECT tick_timestamp,health_state,anomaly_score,maintenance_level FROM spindle_predictions WHERE machine_id=? AND tick_timestamp BETWEEN ? AND ? ORDER BY tick_timestamp", (a["machine_id"], lo, hi)).fetchall()]
    c.close(); return rows


@app.get("/api/regression-tests")
def regression_tests(user: User = Depends(current_user)):
    c = _connect(); rows = [dict(r) for r in c.execute(
        """SELECT id,machine_id,description,start_ts AS start,end_ts AS end,
                  minimum_anomaly_risk,source_alert_id,target_prediction_id,target_timestamp,
                  created_at,created_by,disabled_at,disabled_by
           FROM regression_tests ORDER BY disabled_at IS NOT NULL,created_at DESC"""
    ).fetchall()]; c.close(); return rows


@app.post("/api/regression-tests")
def add_regression(body: RegressionBody, admin: User = Depends(require_admin)):
    if not 0 <= body.minimum_anomaly_risk <= 1: raise HTTPException(400, "minimum_anomaly_risk must be in [0,1]")
    description = body.description.strip()
    if not description or len(description) > 500: raise HTTPException(400, "description must contain 1 to 500 characters")
    try:
        start = datetime.fromisoformat(body.start.replace("Z", "+00:00"))
        end = datetime.fromisoformat(body.end.replace("Z", "+00:00"))
    except ValueError: raise HTTPException(400, "start and end must be ISO-8601 timestamps")
    if start >= end: raise HTTPException(400, "end must be later than start")
    if body.machine_id not in MACHINE_IDS: raise HTTPException(400, "machine_id is not commissioned in the active model")
    c = _connect(); cur = c.execute("INSERT INTO regression_tests(machine_id,description,start_ts,end_ts,minimum_anomaly_risk,created_at,created_by) VALUES(?,?,?,?,?,?,?)", (body.machine_id, description, _iso(start), _iso(end), body.minimum_anomaly_risk, _iso(_now()), admin.username)); c.commit(); rid = cur.lastrowid; c.close(); return {"id": rid}


@app.delete("/api/regression-tests/{regression_id}")
def disable_regression(regression_id: int, admin: User = Depends(require_admin)):
    c = _connect(); cur = c.execute(
        "UPDATE regression_tests SET disabled_at=?,disabled_by=? WHERE id=? AND disabled_at IS NULL",
        (_iso(_now()), admin.username, regression_id),
    )
    if cur.rowcount != 1:
        c.rollback(); c.close(); raise HTTPException(404, "Active regression test not found")
    c.commit(); c.close(); return {"id": regression_id, "disabled": True}


@app.post("/api/backfill")
def backfill(body: BackfillBody, admin: User = Depends(require_admin)):
    return {
        "mode": body.mode,"start":body.start,"end":body.end,
        "processed":0,"skipped_reviewed":0,"machine_ids":[],
        "mock_mode":True,"note":"No production backfill is performed in mock mode.",
    }


@app.get("/api/retrain/status")
def retrain_status(user: User = Depends(current_user)):
    c = _connect()
    grouped = c.execute(
        """SELECT a.machine_id,count(*) AS pending
           FROM reference_candidates rc JOIN alerts a ON a.id=rc.alert_id
           WHERE rc.added_to_reference_at IS NULL
             AND NOT EXISTS (
               SELECT 1 FROM model_version_candidates mvc JOIN model_versions mv ON mv.version_id=mvc.version_id
               WHERE mvc.candidate_id=rc.id AND mv.status='shadow')
           GROUP BY a.machine_id"""
    ).fetchall()
    by_machine = {machine_id: {"pending": 0, "oldest_age_days": 0, "eligible": True, "requires_commissioning": False} for machine_id in MACHINE_IDS}
    for row in grouped:
        by_machine[row["machine_id"]] = {"pending": int(row["pending"]), "oldest_age_days": 0, "eligible": True, "requires_commissioning": False}
    count = sum(item["pending"] for item in by_machine.values())
    active = c.execute("SELECT * FROM retrain_jobs WHERE status IN ('queued','running') ORDER BY created_at DESC LIMIT 1").fetchone()
    latest = c.execute("SELECT * FROM retrain_jobs WHERE status NOT IN ('queued','running') ORDER BY finished_at DESC LIMIT 1").fetchone()
    pending_shadow = c.execute("SELECT version_id,created_at FROM model_versions WHERE status='shadow' ORDER BY created_at DESC LIMIT 1").fetchone()
    c.close(); batch = int(runtime_config.get("RETRAIN_BATCH_SIZE", 50))
    return {"pending": count, "candidate_count": count, "eligible_pending": count, "pending_by_machine": by_machine,
            "batch_size_per_machine": batch, "batch_size": batch,
            "time_cap_days": runtime_config.get("RETRAIN_TIME_CAP_DAYS", 30),
            "due": any(item["pending"] >= batch for item in by_machine.values()),
            "active_job": _row_dict(active) if active else None,
            "last_job": _row_dict(latest) if latest else None,
            "pending_shadow": _row_dict(pending_shadow) if pending_shadow else None,
            "next_check_at": None, "mock_mode": True}


@app.get("/api/retrain/jobs")
def retrain_job_history(limit: int = 20, user: User = Depends(current_user)):
    c = _connect(); rows = [_row_dict(row) for row in c.execute("SELECT * FROM retrain_jobs ORDER BY created_at DESC LIMIT ?", (max(1, min(limit, 100)),)).fetchall()]; c.close(); return rows


@app.get("/api/retrain/jobs/{job_id}")
def retrain_job(job_id: int, user: User = Depends(current_user)):
    c = _connect(); row = c.execute("SELECT * FROM retrain_jobs WHERE id=?", (job_id,)).fetchone(); c.close()
    if not row: raise HTTPException(404, "Retraining job not found")
    return _row_dict(row)


@app.post("/api/retrain/trigger", status_code=202)
def retrain_now(admin: User = Depends(require_admin)):
    vid = f"mock-shadow-{time.time_ns()}"
    report = {
        "mode": "mock",
        "passed": True,
        "note": "Synthetic shadow model created for UI testing only.",
        "retrain_protocol_hash": "mock-demo",
        "false_positive_evaluation": "independent_machine_balanced_holdout",
        "holdout_fraction": artifact_utils.VALIDATION_HOLDOUT_FRACTION,
        "holdout_lineage": "commissioning_forward_holdout",
        "reference_fp_by_machine": {
            machine_id: {
                "holdout_rows": artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS,
                "active_reference_fp": 0.0,
                "shadow_reference_fp": 0.0,
                "maximum_allowed_increase": runtime_config.get("RETRAIN_MAX_FP_RATE_INCREASE", 0.02),
                "pass": True,
            }
            for machine_id in MACHINE_IDS
        },
        "regression_tests": [],
        "reference_fp_gate_pass": True,
        "regression_gate_pass": True,
    }
    c = _connect()
    pending_shadow = c.execute("SELECT version_id,created_at FROM model_versions WHERE status='shadow' ORDER BY created_at DESC LIMIT 1").fetchone()
    if pending_shadow:
        c.close(); return {"queued": False, "reason": "shadow_awaiting_decision", "model": _row_dict(pending_shadow)}
    active = c.execute("SELECT * FROM retrain_jobs WHERE status IN ('queued','running') LIMIT 1").fetchone()
    if active:
        c.close(); return {"queued": False, "reason": "already_running", "job": _row_dict(active)}
    candidate_ids = [int(row[0]) for row in c.execute("SELECT id FROM reference_candidates WHERE added_to_reference_at IS NULL ORDER BY id").fetchall()]
    signature = hashlib.sha256("\n".join(map(str, candidate_ids)).encode()).hexdigest()
    now = _iso(_now())
    cur = c.execute("INSERT INTO retrain_jobs(trigger,requested_by,force,status,candidate_signature,created_at,started_at) VALUES('manual',?,1,'running',?,?,?)", (admin.username, signature, now, now))
    job_id = int(cur.lastrowid)
    c.execute(
        """INSERT INTO model_versions
           (version_id,artifact_path,reference_signature,status,validation_report,created_at,promoted_at,promoted_by)
           VALUES(?,?,?,?,?,?,?,?)""",
        (vid, f"mock://artifacts/{vid}", "mock-reference", "shadow", json.dumps(report), _iso(_now()), None, None),
    )
    for candidate_id in candidate_ids:
        c.execute("INSERT OR IGNORE INTO model_version_candidates(version_id,candidate_id) VALUES(?,?)", (vid, candidate_id))
    result = {"started": True, "version_id": vid, "status": "shadow", "validation": report}
    c.execute("UPDATE retrain_jobs SET status='passed',model_version_id=?,result=?,finished_at=? WHERE id=?", (vid, json.dumps(result), _iso(_now()), job_id))
    c.commit(); job = _row_dict(c.execute("SELECT * FROM retrain_jobs WHERE id=?", (job_id,)).fetchone()); c.close()
    return {"queued": True, "job": job}


@app.get("/api/models")
def models(user: User = Depends(current_user)):
    c = _connect(); rows = [_row_dict(r) for r in c.execute("SELECT * FROM model_versions ORDER BY created_at DESC").fetchall()]; c.close(); return rows


@app.post("/api/models/{version_id}/promote")
def promote(version_id: str, admin: User = Depends(require_admin)):
    c = _connect(); row = c.execute("SELECT status,validation_report FROM model_versions WHERE version_id=?", (version_id,)).fetchone()
    if not row: c.close(); raise HTTPException(404, "Model version not found")
    if row["status"] == "rejected":
        c.close(); raise HTTPException(400, "A rejected model cannot be promoted")
    report = json.loads(row["validation_report"] or "{}")
    if row["status"] == "shadow" and report.get("passed") is not True:
        c.close(); raise HTTPException(400, "A shadow model must pass validation before promotion")
    c.execute("UPDATE model_versions SET status='retired' WHERE status='active' AND version_id<>?", (version_id,))
    c.execute("UPDATE model_versions SET status='active',promoted_at=?,promoted_by=? WHERE version_id=?", (_iso(_now()), admin.username, version_id))
    c.execute(
        """UPDATE reference_candidates SET added_to_reference_at=?
           WHERE id IN (SELECT candidate_id FROM model_version_candidates WHERE version_id=?)""",
        (_iso(_now()), version_id),
    )
    c.commit(); c.close(); return {"active": version_id}


@app.post("/api/models/{version_id}/rollback")
def rollback(version_id: str, admin: User = Depends(require_admin)):
    return promote(version_id, admin)


@app.delete("/api/models/{version_id}")
def delete_model(version_id: str, admin: User = Depends(require_admin)):
    c = _connect()
    row = c.execute("SELECT status FROM model_versions WHERE version_id=?", (version_id,)).fetchone()
    if not row: c.close(); raise HTTPException(404, "Model version not found")
    if row["status"] == "active":
        c.close(); raise HTTPException(400, "Cannot delete the active model version — promote a different version first.")
    c.execute("DELETE FROM model_version_candidates WHERE version_id=?", (version_id,))
    c.execute("DELETE FROM machine_model_calibrations WHERE version_id=?", (version_id,))
    c.execute("DELETE FROM model_calibrations WHERE version_id=?", (version_id,))
    c.execute("DELETE FROM model_versions WHERE version_id=?", (version_id,))
    c.commit(); c.close()
    return {"deleted": version_id}


def _mock_near_miss_ids(c, machine_id: str, hours: float) -> set[int]:
    """Mirror production's timestamp-window slope test for demo data."""
    rows = c.execute(
        "SELECT id,tick_timestamp,anomaly_score,maintenance_level FROM spindle_predictions WHERE machine_id=? ORDER BY tick_timestamp",
        (machine_id,),
    ).fetchall()
    timeline = [
        (row, datetime.fromisoformat(row["tick_timestamp"].replace("Z", "+00:00")))
        for row in rows
    ]
    eligible: set[int] = set()
    window_seconds = float(hours) * 3600.0
    for index, (row, current) in enumerate(timeline):
        if row["maintenance_level"] != "OK" or row["anomaly_score"] is None:
            continue
        samples = []
        for earlier, timestamp in reversed(timeline[:index + 1]):
            elapsed = (current - timestamp).total_seconds()
            if elapsed > window_seconds:
                break
            if earlier["anomaly_score"] is not None:
                samples.append((timestamp.timestamp(), float(earlier["anomaly_score"])))
        if len(samples) < 2:
            continue
        mean_x = sum(x for x, _ in samples) / len(samples)
        mean_y = sum(y for _, y in samples) / len(samples)
        denominator = sum((x - mean_x) ** 2 for x, _ in samples)
        if denominator <= 0:
            continue
        slope = sum((x - mean_x) * (y - mean_y) for x, y in samples) / denominator
        if slope < 0:
            eligible.add(int(row["id"]))
    return eligible


@app.get("/api/history")
def history(limit: int = 50, offset: int = 0, level: str | None = None, machine_id: str = "MACHINE-001", user: User = Depends(current_user)):
    limit = max(1, min(limit, 500)); offset = max(0, offset)
    where = " WHERE machine_id=?"; args: list = [machine_id]
    if level:
        if level not in ("OK", "WARN", "CRITICAL"): raise HTTPException(400, "level must be OK, WARN, or CRITICAL")
        where += " AND maintenance_level=?"; args.append(level)
    c = _connect()
    near_miss_ids = _mock_near_miss_ids(
        c, machine_id, float(runtime_config.get("NEAR_MISS_TREND_WINDOW_HOURS", 6))
    )
    total = c.execute(f"SELECT count(*) FROM spindle_predictions{where}", args).fetchone()[0]
    rows = [_row_dict(r) for r in c.execute(f"SELECT * FROM spindle_predictions{where} ORDER BY tick_timestamp DESC LIMIT ? OFFSET ?", args + [limit, offset]).fetchall()]
    for row in rows:
        alert = c.execute(
            "SELECT status, level, trigger, reviewed_by, reviewed_at FROM alerts WHERE machine_id=? AND tick_timestamp=? AND model_version=? ORDER BY id DESC LIMIT 1",
            (row["machine_id"], row["tick_timestamp"], row["model_version"]),
        ).fetchone()
        row["alert_status"] = alert["status"] if alert else None
        row["alert_level"] = alert["level"] if alert else None
        row["alert_trigger"] = alert["trigger"] if alert else None
        row["alert_reviewed_by"] = alert["reviewed_by"] if alert else None
        row["alert_reviewed_at"] = alert["reviewed_at"] if alert else None
        nmr = c.execute(
            "SELECT status, reviewed_by, reviewed_at FROM near_miss_reviews WHERE prediction_id=?", (row["id"],)
        ).fetchone()
        if nmr:
            row["near_miss_status"] = nmr["status"]
        else:
            # Same eligibility test /api/near-miss uses. Without this, a
            # near-miss that simply hasn't been reviewed yet (no row in
            # near_miss_reviews) looks identical to "never was a near-miss"
            # here — both None — and silently disappears from History
            # instead of showing "Near miss - Pending".
            eligible = int(row["id"]) in near_miss_ids
            row["near_miss_status"] = "pending" if eligible else None
        row["near_miss_reviewed_by"] = nmr["reviewed_by"] if nmr else None
        row["near_miss_reviewed_at"] = nmr["reviewed_at"] if nmr else None
    c.close(); return {"items": rows, "total": total, "limit": limit, "offset": offset}


@app.get("/api/near-miss")
def near_miss(hours: float | None = None, limit: int = 50, offset: int = 0, status: str = "pending", machine_id: str = "MACHINE-001", user: User = Depends(current_user)):
    if status not in ("pending", "acknowledged", "flagged"): raise HTTPException(400, "status must be pending, acknowledged, or flagged")
    hours = float(hours if hours is not None else runtime_config.get("NEAR_MISS_TREND_WINDOW_HOURS", 6))
    if not 0.25 <= hours <= 168: raise HTTPException(400, "hours must be between 0.25 and 168")
    limit = max(1, min(limit, 500)); offset = max(0, offset)
    c = _connect()
    eligible_ids = _mock_near_miss_ids(c, machine_id, hours)
    rows = [_row_dict(r) for r in c.execute("SELECT * FROM spindle_predictions WHERE machine_id=? AND maintenance_level='OK' ORDER BY health_state ASC", (machine_id,)).fetchall() if int(r["id"]) in eligible_ids]
    out = []
    for row in rows:
        review = c.execute("SELECT status, reviewed_by, reviewed_at FROM near_miss_reviews WHERE prediction_id=?", (row["id"],)).fetchone()
        row["review_status"] = review["status"] if review else "pending"
        row["reviewed_by"] = review["reviewed_by"] if review else None
        row["reviewed_at"] = review["reviewed_at"] if review else None
        if row["review_status"] == status: out.append(row)
    c.close()
    total = len(out)
    return {"items": out[offset:offset + limit], "total": total, "limit": limit, "offset": offset}


@app.post("/api/near-miss/{prediction_id}/review")
def review_near_miss(prediction_id: int, body: ReviewBody, user: User = Depends(current_user)):
    if body.decision not in ("acknowledged", "flagged"): raise HTTPException(400, "decision must be acknowledged or flagged")
    with _LOCK:
        c = _connect()
        pred = c.execute("SELECT id, machine_id, tick_timestamp, health_state FROM spindle_predictions WHERE id=?", (prediction_id,)).fetchone()
        if not pred: c.close(); raise HTTPException(404, "Near-miss record not found")
        c.execute(
            """INSERT INTO near_miss_reviews(prediction_id,status,reviewed_by,reviewed_at)
               VALUES(?,?,?,datetime('now'))
               ON CONFLICT(prediction_id) DO UPDATE SET status=excluded.status,reviewed_by=excluded.reviewed_by,reviewed_at=excluded.reviewed_at""",
            (prediction_id, body.decision, user.username),
        )
        # See api/main.py review_near_miss for why: a "flagged" near-miss is
        # a suspected false negative, so it feeds the same regression_tests
        # gate that already guards Alert review's confirmed-normal path in
        # the opposite direction. Mock mode has no real retraining pipeline
        # to gate, but the record is still created so the reviewer workflow
        # and the regression-tests list behave the same as production.
        regression_test_id = None
        if body.decision == "flagged":
            hours = float(runtime_config.get("NEAR_MISS_REGRESSION_WINDOW_HOURS", 1))
            center = datetime.fromisoformat(pred["tick_timestamp"].replace("Z", "+00:00"))
            lo = _iso(center - timedelta(hours=hours)); hi = _iso(center + timedelta(hours=hours))
            health_state = pred["health_state"]
            min_risk = runtime_config.regression_risk_floor(health_state)
            description = f"Near-miss prediction #{prediction_id}"
            existing = c.execute("SELECT id FROM regression_tests WHERE description=? AND disabled_at IS NULL LIMIT 1", (description,)).fetchone()
            if existing:
                regression_test_id = existing["id"]
            else:
                cur = c.execute(
                    """INSERT INTO regression_tests
                       (machine_id,description,start_ts,end_ts,minimum_anomaly_risk,created_at,created_by,
                        target_prediction_id,target_timestamp) VALUES(?,?,?,?,?,?,?,?,?)""",
                    (pred["machine_id"], description, lo, hi, min_risk, _iso(_now()), user.username,
                     prediction_id,pred["tick_timestamp"]),
                )
                regression_test_id = cur.lastrowid
        c.commit(); c.close()
    return {"prediction_id": prediction_id, "status": body.decision, "regression_test_id": regression_test_id}


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    global _live_index
    try: session_user(ws.cookies.get(COOKIE))
    except HTTPException:
        await ws.close(code=4401); return
    machine_id = ws.query_params.get("machine_id", MACHINE_IDS[0])
    if machine_id not in MACHINE_IDS:
        await ws.close(code=4400); return
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(1.0)
            with _LOCK:
                index = _live_index.get(machine_id, 0)
                row = _prediction(index, machine_id)
                c = _connect(); _insert_prediction(c, row, create_alert=(row["maintenance_level"] != "OK" and index % 3 == 0)); c.commit(); c.close(); _live_index[machine_id] = index + 1
            payload = {
                "machine_id": machine_id, "timestamp": row["tick_timestamp"],
                "prediction_timestamp": row["tick_timestamp"], "prediction_lag_seconds": 0.0,
                "model_version": row["model_version"],
                "health_state": row["health_state"], "anomaly_score": row["anomaly_score"],
                "maintenance": {"level": row["maintenance_level"], "reason": row["maintenance_reason"], "trigger": row["maintenance_trigger"]},
                "prediction_available": True, "operating_state": "RUNNING",
                "operating_state_reason": "Synthetic vibration is in the running regime.",
                "operating_state_confidence": 1.0,
                **row["raw_reading"], "mock_mode": True,
            }
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    initialize_mock_database(reset=False)
    print(DB_PATH)
