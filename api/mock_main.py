"""Temporary local demo API backed by SQLite.

Run via `start_project.ps1 -Mock`. This module intentionally bypasses the
production PostgreSQL/ML worker so the web interface can be verified on a
machine that has no database server or trained artifacts yet.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import random
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
import env_manager
import runtime_config
from api.auth import User, current_user, hash_password, issue_cookie, require_admin, verify_password, COOKIE

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "mock_demo.db"
_LOCK = threading.RLock()
_RNG = random.Random(42)
_login_attempts = defaultdict(deque)
_live_index = 0

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


def _maintenance_for(health: float, anomaly: float) -> tuple[str, str, str]:
    if health <= runtime_config.get("FAILURE_HEALTH_THRESHOLD", 20):
        return "CRITICAL", "Health is below the failure threshold.", "health_threshold"
    if health <= runtime_config.get("MAINTENANCE_HEALTH_INSPECT", 30):
        return "WARN", "Health is below the inspection threshold.", "health_inspect"
    if anomaly >= runtime_config.get("MAINTENANCE_PROB_PLAN", 0.60):
        return "WARN", "Mock trend probability indicates inspection should be planned.", "trend_probability"
    return "OK", "No maintenance action is currently required.", "none"


def _sensor_values(i: int) -> dict[str, float]:
    # Slowly wandering, deterministic demo signal with occasional stronger vibration.
    phase = i / 11.0
    bump = 1.3 if i % 37 in (0, 1, 2) else 0.0
    return {
        "a_rms_mps2": round(1.15 + 0.28 * math.sin(phase) + bump, 4),
        "v_rms_mms": round(2.25 + 0.45 * math.sin(phase / 1.4) + bump * 0.7, 4),
        "a_peak_mps2": round(3.8 + 0.9 * math.sin(phase / 0.8) + bump * 2.2, 4),
        "crest_factor": round(3.0 + 0.25 * math.cos(phase) + bump * 0.18, 4),
        "temperature_c": round(37.5 + 2.1 * math.sin(phase / 2.0) + bump * 1.8, 3),
    }


def _prediction(i: int, when: datetime | None = None) -> dict:
    when = when or _now()
    raw = _sensor_values(i)
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
    level, reason, trigger = _maintenance_for(health, anomaly)
    return {
        "tick_timestamp": _iso(when),
        "model_version": "mock-demo-v1",
        "raw_reading": raw,
        "anomaly_score": round(anomaly, 4),
        "health_raw": round(health + _RNG.uniform(-2.0, 2.0), 3),
        "health_state": round(health, 3),
        "trend_slope_per_day": round(-0.08 * math.sin(i / 15.0), 5),
        "remaining_days": round(max(1.0, health / 3.5), 2),
        "failure_probability": {"6h": round(anomaly * 0.40, 4), "12h": round(anomaly * 0.60, 4), "24h": round(anomaly * 0.80, 4)},
        "maintenance_level": level,
        "maintenance_reason": reason,
        "maintenance_trigger": trigger,
        "top_contributors": ["a_rms_mps2", "v_rms_mms"] if level != "OK" else ["temperature_c"],
    }


def _insert_prediction(c: sqlite3.Connection, row: dict, create_alert: bool = False) -> int:
    raw_json = json.dumps(row["raw_reading"])
    c.execute(
        """INSERT INTO spindle_predictions(
        tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,
        trend_slope_per_day,remaining_days,failure_probability,maintenance_level,
        maintenance_reason,maintenance_trigger,top_contributors,is_backfill)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,0)""",
        (
            row["tick_timestamp"], row["model_version"], raw_json, row["anomaly_score"], row["health_raw"],
            row["health_state"], row["trend_slope_per_day"], row["remaining_days"],
            json.dumps(row["failure_probability"]), row["maintenance_level"], row["maintenance_reason"],
            row["maintenance_trigger"], json.dumps(row["top_contributors"]),
        ),
    )
    pid = int(c.execute("SELECT last_insert_rowid()").fetchone()[0])
    if create_alert and row["maintenance_level"] in ("WARN", "CRITICAL"):
        c.execute(
            """INSERT INTO alerts(tick_timestamp,model_version,trigger,level,health_state,anomaly_score,raw_reading,status)
               VALUES(?,?,?,?,?,?,?,'pending')""",
            (row["tick_timestamp"], row["model_version"], row["maintenance_trigger"], row["maintenance_level"],
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
              id INTEGER PRIMARY KEY AUTOINCREMENT, tick_timestamp TEXT NOT NULL, model_version TEXT NOT NULL,
              trigger TEXT NOT NULL, level TEXT NOT NULL, health_state REAL NOT NULL, anomaly_score REAL NOT NULL,
              raw_reading TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending', reviewed_by TEXT, reviewed_at TEXT
            );
            CREATE TABLE IF NOT EXISTS reference_candidates(
              id INTEGER PRIMARY KEY AUTOINCREMENT, alert_id INTEGER UNIQUE, tick_timestamp TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS model_versions(
              version_id TEXT PRIMARY KEY, artifact_path TEXT NOT NULL, reference_signature TEXT NOT NULL,
              status TEXT NOT NULL, validation_report TEXT, created_at TEXT NOT NULL, promoted_at TEXT, promoted_by TEXT
            );
            CREATE TABLE IF NOT EXISTS regression_tests(
              id INTEGER PRIMARY KEY AUTOINCREMENT, description TEXT NOT NULL, start_ts TEXT NOT NULL, end_ts TEXT NOT NULL,
              minimum_anomaly_risk REAL NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS mock_env(key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS spindle_predictions(
              id INTEGER PRIMARY KEY AUTOINCREMENT, tick_timestamp TEXT NOT NULL, model_version TEXT NOT NULL,
              raw_reading TEXT NOT NULL, anomaly_score REAL NOT NULL, health_raw REAL NOT NULL, health_state REAL NOT NULL,
              trend_slope_per_day REAL NOT NULL, remaining_days REAL NOT NULL, failure_probability TEXT NOT NULL,
              maintenance_level TEXT NOT NULL, maintenance_reason TEXT NOT NULL, maintenance_trigger TEXT NOT NULL,
              top_contributors TEXT, is_backfill INTEGER NOT NULL DEFAULT 0
            );
            """
        )
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
                "FRONTEND_ORIGIN": "http://localhost:5173",
            }
            c.executemany("INSERT INTO mock_env(key,value) VALUES(?,?)", vals.items())
        if c.execute("SELECT count(*) FROM model_versions").fetchone()[0] == 0:
            now = _iso(_now())
            c.execute(
                "INSERT INTO model_versions VALUES(?,?,?,?,?,?,?,?)",
                ("mock-demo-v1", "mock://artifacts/mock-demo-v1", "mock-reference", "active", json.dumps({"mode":"mock","passed":True}), now, now, "bootstrap"),
            )
            c.execute(
                "INSERT INTO model_versions VALUES(?,?,?,?,?,?,?,?)",
                ("mock-shadow-v2", "mock://artifacts/mock-shadow-v2", "mock-reference-v2", "shadow", json.dumps({"mode":"mock","passed":True,"note":"demo shadow model"}), now, None, None),
            )
        if c.execute("SELECT count(*) FROM spindle_predictions").fetchone()[0] == 0:
            start = _now() - timedelta(minutes=119)
            for i in range(120):
                row = _prediction(i, start + timedelta(minutes=i))
                _insert_prediction(c, row, create_alert=(row["maintenance_level"] != "OK" and i % 3 == 0))
            _live_index = 120
        else:
            _live_index = int(c.execute("SELECT count(*) FROM spindle_predictions").fetchone()[0])
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
    for key in ("raw_reading", "failure_probability", "top_contributors", "validation_report"):
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
class EnvBody(BaseModel): values: dict[str, str]
class UserCreate(BaseModel): username: str; password: str; role: str = "viewer"
class RegressionBody(BaseModel): description: str; start: str; end: str; minimum_anomaly_risk: float = 0.6
class BackfillBody(BaseModel): start: str; end: str; mode: str = "repredict"


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


@app.post("/api/users")
def create_user(body: UserCreate, admin: User = Depends(require_admin)):
    if body.role not in ("viewer", "admin"): raise HTTPException(400, "role must be viewer or admin")
    c = _connect()
    try:
        c.execute("INSERT INTO app_users(username,password_hash,role) VALUES(?,?,?)", (body.username, hash_password(body.password), body.role)); c.commit()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(400, str(exc))
    finally: c.close()
    return {"username": body.username, "role": body.role}


@app.get("/api/env")
def get_env(admin: User = Depends(require_admin)):
    c = _connect(); rows = c.execute("SELECT key,value FROM mock_env ORDER BY key").fetchall(); c.close()
    out = {r["key"]: r["value"] for r in rows}
    if out.get("PG_PASSWORD"): out["PG_PASSWORD"] = env_manager.MASK
    return out


@app.put("/api/env")
def put_env(body: EnvBody, admin: User = Depends(require_admin)):
    c = _connect(); changed = []
    for key, value in body.values.items():
        if key == "PG_PASSWORD" and value == env_manager.MASK: continue
        old = c.execute("SELECT value FROM mock_env WHERE key=?", (key,)).fetchone()
        if old is None or old["value"] != str(value):
            c.execute("INSERT INTO mock_env(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (key, str(value)))
            changed.append(key)
    c.commit(); c.close()
    return {"changed_keys": changed, "worker_restart_requested": False, "mock_mode": True}


@app.get("/api/config/thresholds")
def thresholds(user: User = Depends(current_user)):
    return {k: runtime_config.get(k) for k in ("MAINTENANCE_PROB_URGENT", "MAINTENANCE_PROB_PLAN", "FAILURE_HEALTH_THRESHOLD", "MAINTENANCE_HEALTH_INSPECT")}


@app.put("/api/config/thresholds")
def set_thresholds(body: ThresholdBody, admin: User = Depends(require_admin)):
    values = runtime_config.validate_thresholds(body.model_dump())
    c = _connect()
    for key, value in values.items():
        c.execute("INSERT INTO runtime_config(key,value,updated_by) VALUES(?,?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value,updated_by=excluded.updated_by", (key, json.dumps(value), admin.username))
    c.commit(); c.close(); runtime_config.set_local(values)
    return values


@app.get("/api/alerts")
def alerts(status: str = "pending", trigger: str | None = None, limit: int = 100, offset: int = 0, user: User = Depends(current_user)):
    sql = "SELECT * FROM alerts WHERE status=?"; args: list = [status]
    if trigger: sql += " AND trigger=?"; args.append(trigger)
    sql += " ORDER BY tick_timestamp DESC LIMIT ? OFFSET ?"; args.extend([max(1, min(limit, 500)), max(0, offset)])
    c = _connect(); rows = [_row_dict(r) for r in c.execute(sql, args).fetchall()]; c.close(); return rows


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
def alert_context(alert_id: int, hours: int = 3, user: User = Depends(current_user)):
    c = _connect(); a = c.execute("SELECT tick_timestamp FROM alerts WHERE id=?", (alert_id,)).fetchone()
    if not a: c.close(); raise HTTPException(404, "Alert not found")
    center = datetime.fromisoformat(a["tick_timestamp"].replace("Z", "+00:00")); lo = _iso(center - timedelta(hours=hours)); hi = _iso(center + timedelta(hours=hours))
    rows = [dict(r) for r in c.execute("SELECT tick_timestamp,health_state,anomaly_score,maintenance_level FROM spindle_predictions WHERE tick_timestamp BETWEEN ? AND ? ORDER BY tick_timestamp", (lo, hi)).fetchall()]
    c.close(); return rows


@app.get("/api/regression-tests")
def regression_tests(user: User = Depends(current_user)):
    c = _connect(); rows = [dict(r) for r in c.execute("SELECT * FROM regression_tests ORDER BY created_at DESC").fetchall()]; c.close(); return rows


@app.post("/api/regression-tests")
def add_regression(body: RegressionBody, admin: User = Depends(require_admin)):
    if not 0 <= body.minimum_anomaly_risk <= 1: raise HTTPException(400, "minimum_anomaly_risk must be in [0,1]")
    c = _connect(); cur = c.execute("INSERT INTO regression_tests(description,start_ts,end_ts,minimum_anomaly_risk,created_at) VALUES(?,?,?,?,?)", (body.description, body.start, body.end, body.minimum_anomaly_risk, _iso(_now()))); c.commit(); rid = cur.lastrowid; c.close(); return {"id": rid}


@app.post("/api/backfill")
def backfill(body: BackfillBody, admin: User = Depends(require_admin)):
    return {"mode": body.mode, "start": body.start, "end": body.end, "rows": 0, "mock_mode": True, "note": "No production backfill is performed in mock mode."}


@app.get("/api/retrain/status")
def retrain_status(user: User = Depends(current_user)):
    c = _connect(); count = c.execute("SELECT count(*) FROM reference_candidates").fetchone()[0]; c.close()
    return {"candidate_count": count, "batch_size": runtime_config.get("RETRAIN_BATCH_SIZE", 50), "due": count >= runtime_config.get("RETRAIN_BATCH_SIZE", 50), "mock_mode": True}


@app.post("/api/retrain/trigger")
def retrain_now(admin: User = Depends(require_admin)):
    vid = f"mock-shadow-{int(time.time())}"
    report = {"mode": "mock", "passed": True, "note": "Synthetic shadow model created for UI testing only."}
    c = _connect(); c.execute("INSERT INTO model_versions VALUES(?,?,?,?,?,?,?,?)", (vid, f"mock://artifacts/{vid}", "mock-reference", "shadow", json.dumps(report), _iso(_now()), None, None)); c.commit(); c.close()
    return {"version_id": vid, "status": "shadow", "validation_report": report}


@app.get("/api/models")
def models(user: User = Depends(current_user)):
    c = _connect(); rows = [_row_dict(r) for r in c.execute("SELECT * FROM model_versions ORDER BY created_at DESC").fetchall()]; c.close(); return rows


@app.post("/api/models/{version_id}/promote")
def promote(version_id: str, admin: User = Depends(require_admin)):
    c = _connect(); row = c.execute("SELECT status FROM model_versions WHERE version_id=?", (version_id,)).fetchone()
    if not row: c.close(); raise HTTPException(404, "Model version not found")
    c.execute("UPDATE model_versions SET status='retired' WHERE status='active'")
    c.execute("UPDATE model_versions SET status='active',promoted_at=?,promoted_by=? WHERE version_id=?", (_iso(_now()), admin.username, version_id)); c.commit(); c.close(); return {"active": version_id}


@app.post("/api/models/{version_id}/rollback")
def rollback(version_id: str, admin: User = Depends(require_admin)):
    return promote(version_id, admin)


@app.get("/api/history")
def history(limit: int = 200, user: User = Depends(current_user)):
    c = _connect(); rows = [_row_dict(r) for r in c.execute("SELECT * FROM spindle_predictions ORDER BY tick_timestamp DESC LIMIT ?", (max(1, min(limit, 1000)),)).fetchall()]; c.close(); return rows


@app.get("/api/near-miss")
def near_miss(hours: int = 6, limit: int = 50, user: User = Depends(current_user)):
    c = _connect(); rows = [_row_dict(r) for r in c.execute("SELECT * FROM spindle_predictions WHERE maintenance_level='OK' AND anomaly_score BETWEEN 0.18 AND 0.60 ORDER BY anomaly_score DESC LIMIT ?", (max(1, min(limit, 500)),)).fetchall()]; c.close(); return rows


@app.websocket("/ws/live")
async def live(ws: WebSocket):
    global _live_index
    await ws.accept()
    try:
        while True:
            await asyncio.sleep(1.0)
            with _LOCK:
                row = _prediction(_live_index)
                c = _connect(); _insert_prediction(c, row, create_alert=(row["maintenance_level"] != "OK" and _live_index % 3 == 0)); c.commit(); c.close(); _live_index += 1
            payload = {
                "timestamp": row["tick_timestamp"], "model_version": row["model_version"],
                "health_state": row["health_state"], "anomaly_score": row["anomaly_score"],
                "maintenance": {"level": row["maintenance_level"], "reason": row["maintenance_reason"], "trigger": row["maintenance_trigger"]},
                **row["raw_reading"], "mock_mode": True,
            }
            await ws.send_text(json.dumps(payload))
    except WebSocketDisconnect:
        pass


if __name__ == "__main__":
    initialize_mock_database(reset=False)
    print(DB_PATH)
