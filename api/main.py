from __future__ import annotations
import asyncio
import json
import os
import select
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import psycopg2.errors
import psycopg2.extras
from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

import config
import artifact_utils
import db
import db_schema
import env_manager
import model_registry
import retrain_service
import retrain_jobs
import runtime_config
import backfill
from api.auth import COOKIE, User, current_user, hash_password, issue_cookie, require_admin, session_user, verify_password

app=FastAPI(title="Spindle Condition Monitoring API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("FRONTEND_ORIGIN","http://localhost:5173")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
_login_attempts=defaultdict(deque)
scheduler=BackgroundScheduler(daemon=True)


def conn(): return db.get_connection()


def machine(value: str | None) -> str:
    value = (value or db.DEFAULT_MACHINE_ID).strip()
    if not value or len(value) > 128:
        raise HTTPException(400, "machine_id must contain 1 to 128 characters")
    return value


def _register_current_artifacts(cur):
    """Register a newly trained root bundle and make it active.

    The manual trainer intentionally writes artifacts before the API starts.
    A fresh training metadata file has no version_id; that is the explicit
    signal that this bundle has not yet entered the database registry.
    """
    model_path=os.path.join(config.ARTIFACTS_DIR,"isolation_forest.pkl")
    metadata_path=os.path.join(config.ARTIFACTS_DIR,"metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        return
    with open(metadata_path,encoding="utf-8") as handle:
        metadata=json.load(handle)

    version_id=metadata.get("version_id")
    if version_id:
        cur.execute("SELECT 1 FROM model_versions WHERE version_id=%s",(version_id,))
        if cur.fetchone():
            return
        path=model_registry.bundle_path(version_id)
        if not os.path.isdir(path):
            path=model_registry.snapshot_current(version_id)
    else:
        version_id=model_registry.new_version_id()
        path=model_registry.snapshot_current(version_id)

    # The versioned metadata is stamped by snapshot_current. Reinstall it so
    # the worker reports the same version that is stored in model_versions.
    model_registry.install_bundle(version_id)
    reference_rows=artifact_utils.load_reference_rows()
    if reference_rows is not None:
        signature_source=[f"{row['machine_id']}\0{row['timestamp']}" for row in reference_rows]
    else:
        timestamps=artifact_utils.load_reference_timestamps()
        signature_source=timestamps if timestamps is not None else []
    signature=model_registry.reference_signature(signature_source)

    cur.execute("UPDATE model_versions SET status='retired' WHERE status='active'")
    cur.execute(
        """INSERT INTO model_versions
           (version_id,artifact_path,reference_signature,status,promoted_at,promoted_by)
           VALUES(%s,%s,%s,'active',now(),'manual-training-bootstrap')""",
        (version_id,path,signature),
    )
    for machine_id,item in (artifact_utils.load_machine_calibrations() or {}).items():
        cur.execute(
            """INSERT INTO model_calibrations
               (version_id,machine_id,calibration,source_rows,source_description,created_by)
               VALUES(%s,%s,%s::jsonb,%s,%s,'manual-training-bootstrap') RETURNING id""",
            (
                version_id,machine_id,json.dumps(item["calibration"]),item["source_rows"],
                "Automatic robust machine condition anchor from commissioning training",
            ),
        )
        calibration_id=cur.fetchone()[0]
        cur.execute(
            """INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id)
               VALUES(%s,%s,%s)""",
            (machine_id,version_id,calibration_id),
        )


def _bootstrap():
    c=conn(); db_schema.migrate(c)
    # api/main.py's own process never read persisted runtime_config back in
    # (only worker.py did, on the config_changed NOTIFY) — GETs here just
    # returned in-memory _DEFAULTS until something PUT in THIS process.
    # That's mostly cosmetic for thresholds (worker enforces the real
    # policy either way), but _scheduled_retrain() below runs in this
    # process and needs an accurate AUTO_RETRAIN_ENABLED on every restart,
    # not just "since the last PUT" — so load it here too.
    runtime_config.load_from_db(c)
    with c.cursor() as cur:
        cur.execute("SELECT count(*) FROM app_users"); n=cur.fetchone()[0]
        if n==0:
            username=os.getenv("BOOTSTRAP_ADMIN_USER")
            password=os.getenv("BOOTSTRAP_ADMIN_PASSWORD")
            if username and password:
                if not env_manager.bootstrap_password_is_secure(password):
                    c.rollback()
                    c.close()
                    raise RuntimeError(
                        "BOOTSTRAP_ADMIN_PASSWORD must be a unique password of at least 12 characters"
                    )
                cur.execute("INSERT INTO app_users(username,password_hash,role) VALUES(%s,%s,'admin')", (username,hash_password(password)))
        _register_current_artifacts(cur)
    c.commit(); c.close()


def _scheduled_retrain():
    c=None
    try:
        c=conn()
        queued_ids=retrain_jobs.queued_ids(c)
        if queued_ids:
            for job_id in queued_ids: _schedule_retrain_job(job_id)
            return
        if not runtime_config.get("AUTO_RETRAIN_ENABLED", True):
            return
        due,_=retrain_service.should_retrain(c)
        if not due:
            return
        queued=retrain_jobs.enqueue(c,"scheduled","scheduler",False)
        if queued.get("queued"):
            _schedule_retrain_job(int(queued["job"]["id"]))
    except Exception as e:
        print(f"[retrain-job] {e}")
    finally:
        if c is not None:
            c.close()

def _configure_retrain_job():
    minutes=int(runtime_config.get("RETRAIN_CHECK_INTERVAL_MINUTES",60))
    scheduler.add_job(
        _scheduled_retrain,"interval",minutes=minutes,
        id="retrain-check",replace_existing=True,coalesce=True,max_instances=1,
    )


def _schedule_retrain_job(job_id:int):
    scheduler.add_job(
        retrain_jobs.execute,"date",run_date=datetime.now(timezone.utc),args=[job_id],
        id=f"retrain-job-{job_id}",replace_existing=True,misfire_grace_time=3600,
    )

@app.on_event("startup")
def startup():
    if env_manager.ensure_secure_app_secret():
        print("[security] Replaced an absent or placeholder APP_SECRET_KEY; existing sessions are invalid")
    _bootstrap()
    _configure_retrain_job()
    c=conn()
    try: queued_jobs=retrain_jobs.recover_interrupted(c)
    finally: c.close()
    for job_id in queued_jobs: _schedule_retrain_job(job_id)
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    if scheduler.running:
        scheduler.shutdown(wait=False)

@app.get("/api/health")
def health():
    return {"ok": True, "mode": "production"}


class LoginBody(BaseModel): username:str; password:str
class ReviewBody(BaseModel): decision:str
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
class EnvBody(BaseModel): values: dict[str,str]
class UserCreate(BaseModel): username:str; password:str; role:str="viewer"
class RegressionBody(BaseModel):
    description:str
    start:str
    end:str
    machine_id:str=db.DEFAULT_MACHINE_ID
    minimum_anomaly_risk:float=0.6
class BackfillBody(BaseModel): start:str; end:str; mode:str="repredict"; machine_id:str|None=None
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
def login(body:LoginBody,response:Response):
    now=time.time(); q=_login_attempts[body.username]
    while q and now-q[0]>60: q.popleft()
    if len(q)>=10: raise HTTPException(429,"Too many login attempts; retry shortly")
    q.append(now)
    c=conn()
    with c.cursor() as cur:
        cur.execute("SELECT password_hash,role,disabled FROM app_users WHERE username=%s",(body.username,)); row=cur.fetchone()
    c.close()
    if not row or row[2] or not verify_password(body.password,row[0]): raise HTTPException(401,"Invalid credentials")
    q.clear(); issue_cookie(response,body.username,row[1]); return {"username":body.username,"role":row[1]}

@app.post("/api/logout")
def logout(response:Response): response.delete_cookie(COOKIE); return {"ok":True}
@app.get("/api/me")
def me(user:User=Depends(current_user)): return user.__dict__

@app.get("/api/machines")
def machines(user:User=Depends(current_user)):
    c=conn()
    try:
        # The configured raw source is authoritative. Persisted control-plane
        # rows may contain retired/test machine IDs and must not populate the
        # production selector.
        items=db.fetch_machine_ids(c,db.get_table_name())
    except Exception as exc:
        c.rollback()
        raise HTTPException(503,f"Cannot discover machines from PostgreSQL source {db.get_table_name()!r}: {exc}")
    finally:
        c.close()
    items=sorted(set(items))
    if not items:
        raise HTTPException(503,f"PostgreSQL source {db.get_table_name()!r} contains no machine IDs")
    default=db.DEFAULT_MACHINE_ID if db.DEFAULT_MACHINE_ID in items else items[0]
    return {"items":items,"default":default,"source":"postgresql","table":db.get_table_name()}


@app.get("/api/fleet/condition-trend")
def fleet_condition_trend(days:int=7,user:User=Depends(current_user)):
    if not 1 <= days <= 31:
        raise HTTPException(400,"days must be between 1 and 31")
    end=datetime.now(timezone.utc)
    start=end-timedelta(days=days)
    c=conn()
    try:
        machine_ids=sorted(set(db.fetch_machine_ids(c,db.get_table_name())))
        if not machine_ids:
            raise HTTPException(503,f"PostgreSQL source {db.get_table_name()!r} contains no machine IDs")
        with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """WITH canonical_tick AS (
                     SELECT DISTINCT ON (machine_id,tick_timestamp)
                            machine_id,tick_timestamp,health_state
                     FROM spindle_predictions
                     WHERE tick_timestamp >= %s AND tick_timestamp <= %s
                       AND machine_id = ANY(%s)
                       AND health_state IS NOT NULL
                     ORDER BY machine_id,tick_timestamp,created_at DESC,id DESC
                   )
                   SELECT machine_id,
                          date_trunc('hour',tick_timestamp) AS bucket,
                          avg(health_state) AS health_state,
                          count(*) AS samples
                   FROM canonical_tick
                   GROUP BY machine_id,date_trunc('hour',tick_timestamp)
                   ORDER BY machine_id,bucket""",
                (start,end,machine_ids),
            )
            rows=cur.fetchall()
    except HTTPException:
        raise
    except Exception as exc:
        c.rollback()
        raise HTTPException(503,f"Cannot load fleet condition trend: {exc}")
    finally:
        c.close()
    grouped:dict[str,list[dict]]={}
    for row in rows:
        grouped.setdefault(row["machine_id"],[]).append({
            "timestamp":row["bucket"],
            "health_state":float(row["health_state"]),
            "samples":int(row["samples"]),
        })
    return {
        "days":days,"metric":"hourly_average_condition","start":start,"end":end,
        "series":[{"machine_id":machine_id,"points":grouped.get(machine_id,[])} for machine_id in machine_ids],
    }


@app.get("/api/live/latest")
def latest_live(machine_id:str|None=None,user:User=Depends(current_user)):
    selected_machine=machine(machine_id)
    c=conn()
    try:
        source_row=db.fetch_latest_row(c,db.get_table_name(),selected_machine)
        if source_row is None:
            raise HTTPException(404,f"No source rows found for machine {selected_machine}")
        columns=db.get_db_columns()
        raw=db.canonical_sensor_reading(source_row,columns)
        with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT tick_timestamp,model_version,anomaly_score,health_state,
                          maintenance_level,maintenance_reason,maintenance_trigger
                   FROM spindle_predictions WHERE machine_id=%s
                   ORDER BY tick_timestamp DESC LIMIT 1""",
                (selected_machine,),
            )
            prediction=cur.fetchone()
            cur.execute(
                """SELECT operating_state,reason,confidence,activity_score,stop_threshold,run_threshold,
                          tick_timestamp,state_changed_at
                   FROM machine_runtime_state WHERE machine_id=%s""",
                (selected_machine,),
            )
            runtime_state=cur.fetchone()
    except HTTPException:
        raise
    except Exception as exc:
        c.rollback()
        raise HTTPException(503,f"Cannot read live PostgreSQL source for {selected_machine}: {exc}")
    finally:
        c.close()

    source_timestamp=source_row[columns["timestamp"]]
    source_age_seconds=None
    try:
        timestamp_for_age=source_timestamp
        if timestamp_for_age.tzinfo is None:
            timestamp_for_age=timestamp_for_age.replace(tzinfo=timezone.utc)
        source_age_seconds=max(0.0,(datetime.now(timezone.utc)-timestamp_for_age).total_seconds())
    except (AttributeError,TypeError,ValueError):
        pass

    operating="UNKNOWN"
    operating_reason="The worker has not published an operating state yet."
    operating_confidence=0.0
    state_changed_at=None
    runtime_tick=None
    if runtime_state:
        operating=runtime_state["operating_state"]
        operating_reason=runtime_state["reason"]
        operating_confidence=runtime_state["confidence"]
        state_changed_at=runtime_state["state_changed_at"]
        runtime_tick=runtime_state["tick_timestamp"]
    stale_seconds=runtime_config.get("SOURCE_STALE_SECONDS",config.SOURCE_STALE_SECONDS)
    if source_age_seconds is not None and source_age_seconds > stale_seconds:
        operating="NO_DATA"
        operating_reason=f"Newest source row is {int(source_age_seconds)} seconds old; waiting for fresh sensor data."
        operating_confidence=1.0

    prediction_lag_seconds=None
    prediction_matches_source=False
    if prediction is not None:
        try:
            prediction_matches_source=prediction["tick_timestamp"] == source_timestamp
            prediction_lag_seconds=max(
                0.0,
                (source_timestamp-prediction["tick_timestamp"]).total_seconds(),
            )
        except (AttributeError,TypeError,ValueError):
            prediction_lag_seconds=None
    prediction_is_current=(
        prediction is not None
        and operating in ("RUNNING","UNKNOWN")
        and prediction_matches_source
    )
    if prediction_is_current and state_changed_at is not None:
        prediction_is_current=prediction["tick_timestamp"] >= state_changed_at
    if prediction_is_current and runtime_tick is not None:
        prediction_is_current=prediction["tick_timestamp"] >= runtime_tick

    response={
        "machine_id":selected_machine,
        "timestamp":source_timestamp,
        "source":"postgresql",
        "source_table":db.get_table_name(),
        "source_age_seconds":source_age_seconds,
        "prediction_lag_seconds":prediction_lag_seconds,
        "prediction_timestamp":prediction["tick_timestamp"] if prediction is not None else None,
        "prediction_available":prediction_is_current,
        "operating_state":operating,
        "operating_state_reason":operating_reason,
        "operating_state_confidence":operating_confidence,
        "operating_state_changed_at":state_changed_at,
        "operating_state_activity":runtime_state["activity_score"] if runtime_state else None,
        "operating_state_stop_threshold":runtime_state["stop_threshold"] if runtime_state else None,
        "operating_state_run_threshold":runtime_state["run_threshold"] if runtime_state else None,
        **raw,
    }
    if prediction is not None and not prediction_is_current and operating in ("RUNNING","UNKNOWN"):
        response["prediction_wait_reason"]=(
            "The source has a newer sensor row than the inference worker. "
            "Condition and maintenance values are hidden until that exact row is processed."
        )
    if prediction_is_current:
        response.update({
            "model_version":prediction["model_version"],
            "anomaly_score":prediction["anomaly_score"],
            "health_state":prediction["health_state"],
            "maintenance":{
                "level":prediction["maintenance_level"],
                "reason":prediction["maintenance_reason"],
                "trigger":prediction["maintenance_trigger"],
            },
        })
    return response

@app.post("/api/users")
def create_user(body:UserCreate,admin:User=Depends(require_admin)):
    if body.role not in ("viewer","admin"): raise HTTPException(400,"role must be viewer or admin")
    username=body.username.strip()
    if not 3<=len(username)<=64: raise HTTPException(400,"username must contain 3 to 64 characters")
    if not env_manager.bootstrap_password_is_secure(body.password): raise HTTPException(400,"password must be a unique password of at least 12 characters")
    c=conn()
    try:
        with c.cursor() as cur: cur.execute("INSERT INTO app_users(username,password_hash,role) VALUES(%s,%s,%s)",(username,hash_password(body.password),body.role))
        c.commit()
    except psycopg2.errors.UniqueViolation: c.rollback(); raise HTTPException(409,"username already exists")
    finally: c.close()
    return {"username":username,"role":body.role}

@app.get("/api/env")
def get_env(admin:User=Depends(require_admin)): return env_manager.read_masked()
@app.put("/api/env")
def put_env(body:EnvBody,admin:User=Depends(require_admin)):
    try: changed=env_manager.write(body.values)
    except ValueError as e: raise HTTPException(400,str(e))
    c=conn()
    with c.cursor() as cur:
        cur.execute("INSERT INTO env_change_log(changed_by,changed_keys) VALUES(%s,%s::jsonb)",(admin.username,json.dumps(changed)))
        cur.execute("NOTIFY env_changed")
    c.commit(); c.close()
    # Future API DB connections use the just-saved local .env values. The
    # worker receives env_changed and reconnects only after its clean restart.
    env_manager.apply_to_process_environment()
    return {"changed_keys":changed,"worker_restart_requested":bool(changed)}

@app.get("/api/config/thresholds")
def thresholds(user:User=Depends(current_user)):
    return {k:runtime_config.get(k) for k in runtime_config.THRESHOLD_KEYS}
@app.put("/api/config/thresholds")
def set_thresholds(body:ThresholdBody,admin:User=Depends(require_admin)):
    try: values=runtime_config.validate_thresholds(body.model_dump())
    except ValueError as e: raise HTTPException(400,str(e))
    c=conn(); runtime_config.save_to_db(c,values,admin.username); c.close(); return values

@app.get("/api/config/training")
def training_config(user:User=Depends(current_user)):
    # Same knobs retrain_service.py already reads via runtime_config.get()
    # (should_retrain(), _dedup_within_machine(), reference balancing) — this
    # just surfaces them for the Models page instead of requiring a direct
    # DB write to change what "training" does. AUTO_RETRAIN_ENABLED gates
    # _scheduled_retrain() (the configurable APScheduler job) directly; it
    # doesn't affect the "Run shadow retrain" button, which is always a
    # deliberate manual action regardless of this setting.
    keys=(*runtime_config.TRAINING_KEYS,"AUTO_RETRAIN_ENABLED")
    return {k:runtime_config.get(k) for k in keys}
@app.put("/api/config/training")
def set_training_config(body:TrainingConfigBody,admin:User=Depends(require_admin)):
    requested=body.model_dump()
    try: values=runtime_config.validate_training_config(requested)
    except ValueError as e: raise HTTPException(400,str(e))
    values["AUTO_RETRAIN_ENABLED"]=body.AUTO_RETRAIN_ENABLED
    c=conn(); runtime_config.save_to_db(c,values,admin.username); c.close()
    _configure_retrain_job()
    return values

@app.get("/api/alerts")
def alerts(status:str="pending",trigger:str|None=None,machine_id:str|None=None,limit:int=50,offset:int=0,user:User=Depends(current_user)):
    limit=max(1,min(limit,500)); offset=max(0,offset); c=conn()
    where=["status=%s","machine_id=%s"]; args=[status,machine(machine_id)]
    if trigger: where.append("trigger=%s"); args.append(trigger)
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM alerts WHERE {' AND '.join(where)}",args); total=cur.fetchone()["total"]
        cur.execute(f"SELECT * FROM alerts WHERE {' AND '.join(where)} ORDER BY tick_timestamp DESC LIMIT %s OFFSET %s",args+[limit,offset]); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.post("/api/alerts/{alert_id}/review")
def review(alert_id:int,body:ReviewBody,user:User=Depends(current_user)):
    if body.decision not in ("confirmed_anomaly","confirmed_normal"): raise HTTPException(400,"decision must be confirmed_anomaly or confirmed_normal")
    c=conn()
    with c.cursor() as cur:
        cur.execute("UPDATE alerts SET status=%s,reviewed_by=%s,reviewed_at=now() WHERE id=%s AND status='pending' RETURNING tick_timestamp",(body.decision,user.username,alert_id)); row=cur.fetchone()
        if not row: c.rollback(); c.close(); raise HTTPException(409,"Alert not found or already reviewed")
        if body.decision=="confirmed_normal":
            cur.execute("INSERT INTO reference_candidates(alert_id,tick_timestamp) VALUES(%s,%s) ON CONFLICT(alert_id) DO NOTHING",(alert_id,row[0]))
    c.commit(); c.close(); return {"id":alert_id,"status":body.decision}


@app.get("/api/alerts/{alert_id}/context")
def alert_context(alert_id:int,hours:float=3,user:User=Depends(current_user)):
    if not 0.25 <= hours <= 168: raise HTTPException(400,"hours must be between 0.25 and 168")
    c=conn()
    with c.cursor() as cur:
        cur.execute("SELECT tick_timestamp,machine_id FROM alerts WHERE id=%s",(alert_id,)); row=cur.fetchone()
        if not row: c.close(); raise HTTPException(404,"Alert not found")
        ts,machine_id=row
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT tick_timestamp,health_state,anomaly_score,maintenance_level FROM spindle_predictions
                       WHERE machine_id=%s AND tick_timestamp BETWEEN %s-(%s||' hours')::interval AND %s+(%s||' hours')::interval
                       ORDER BY tick_timestamp""",(machine_id,ts,hours,ts,hours)); rows=cur.fetchall()
    c.close(); return rows

@app.get("/api/regression-tests")
def regression_tests(user:User=Depends(current_user)):
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT id,machine_id,description,lower(timestamp_range) AS start,
                      upper(timestamp_range) AS end,minimum_anomaly_risk,
                      source_alert_id,target_prediction_id,target_timestamp,
                      created_at,created_by,disabled_at,disabled_by
               FROM regression_tests
               ORDER BY disabled_at NULLS FIRST,created_at DESC"""
        ); rows=cur.fetchall()
    c.close(); return rows

@app.post("/api/regression-tests")
def add_regression(body:RegressionBody,admin:User=Depends(require_admin)):
    if not 0<=body.minimum_anomaly_risk<=1: raise HTTPException(400,"minimum_anomaly_risk must be in [0,1]")
    description=body.description.strip()
    if not description or len(description)>500: raise HTTPException(400,"description must contain 1 to 500 characters")
    try:
        start=datetime.fromisoformat(body.start.replace("Z","+00:00"))
        end=datetime.fromisoformat(body.end.replace("Z","+00:00"))
    except ValueError: raise HTTPException(400,"start and end must be ISO-8601 timestamps")
    if start>=end: raise HTTPException(400,"end must be later than start")
    c=conn()
    with c.cursor() as cur:
        selected_machine=machine(body.machine_id)
        cur.execute(
            """SELECT 1 FROM machine_model_calibrations mmc
               JOIN model_versions mv ON mv.version_id=mmc.version_id
               WHERE mv.status='active' AND mmc.machine_id=%s LIMIT 1""",
            (selected_machine,),
        )
        commissioned=bool(cur.fetchone())
        if commissioned:
            cur.execute("INSERT INTO regression_tests(machine_id,description,timestamp_range,minimum_anomaly_risk,created_by) VALUES(%s,%s,tstzrange(%s,%s,'[]'),%s,%s) RETURNING id",(selected_machine,description,start,end,body.minimum_anomaly_risk,admin.username)); rid=cur.fetchone()[0]
    if not commissioned:
        c.rollback(); c.close(); raise HTTPException(400,"machine_id is not commissioned in the active model")
    c.commit(); c.close(); return {"id":rid}

@app.delete("/api/regression-tests/{regression_id}")
def disable_regression(regression_id:int,admin:User=Depends(require_admin)):
    c=conn()
    with c.cursor() as cur:
        cur.execute(
            """UPDATE regression_tests SET disabled_at=now(),disabled_by=%s
               WHERE id=%s AND disabled_at IS NULL RETURNING id""",
            (admin.username,regression_id),
        )
        row=cur.fetchone()
    if not row:
        c.rollback(); c.close(); raise HTTPException(404,"Active regression test not found")
    c.commit(); c.close(); return {"id":regression_id,"disabled":True}

@app.post("/api/backfill")
def run_backfill(body:BackfillBody,admin:User=Depends(require_admin)):
    c=conn()
    try: return backfill.run(c,body.start,body.end,body.mode,machine(body.machine_id) if body.machine_id else None)
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()

@app.get("/api/retrain/status")
def retrain_status(user:User=Depends(current_user)):
    c=conn()
    try:
        due,status=retrain_service.should_retrain(c)
        active_job,last_job=retrain_jobs.active_and_latest(c)
        pending_shadow=retrain_jobs.pending_shadow(c)
    finally: c.close()
    scheduled=scheduler.get_job("retrain-check")
    return {**status,"due":due,"active_job":active_job,"last_job":last_job,"pending_shadow":pending_shadow,
            "next_check_at":scheduled.next_run_time if scheduled else None}
@app.get("/api/retrain/jobs")
def retrain_job_history(limit:int=20,user:User=Depends(current_user)):
    c=conn()
    try: return retrain_jobs.list_recent(c,max(1,min(limit,100)))
    finally: c.close()
@app.get("/api/retrain/jobs/{job_id}")
def retrain_job(job_id:int,user:User=Depends(current_user)):
    c=conn()
    try: job=retrain_jobs.get(c,job_id)
    finally: c.close()
    if not job: raise HTTPException(404,"Retraining job not found")
    return job
@app.post("/api/retrain/trigger",status_code=202)
def retrain_now(admin:User=Depends(require_admin)):
    c=conn()
    try: queued=retrain_jobs.enqueue(c,"manual",admin.username,True)
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()
    if queued.get("queued"): _schedule_retrain_job(int(queued["job"]["id"]))
    return queued

@app.get("/api/models")
def models(user:User=Depends(current_user)):
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM model_versions ORDER BY created_at DESC"); rows=cur.fetchall()
    c.close(); return rows
@app.post("/api/models/{version_id}/promote")
def promote(version_id:str,admin:User=Depends(require_admin)):
    c=conn()
    try: model_registry.promote(c,version_id,admin.username)
    except (ValueError,FileNotFoundError) as e: raise HTTPException(400,str(e))
    finally: c.close()
    return {"active":version_id}
@app.post("/api/models/{version_id}/rollback")
def rollback(version_id:str,admin:User=Depends(require_admin)): return promote(version_id,admin)
@app.delete("/api/models/{version_id}")
def delete_model(version_id:str,admin:User=Depends(require_admin)):
    c=conn()
    try: model_registry.delete_version(c,version_id)
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()
    return {"deleted":version_id}

@app.get("/api/history")
def history(limit:int=50,offset:int=0,level:str|None=None,machine_id:str|None=None,user:User=Depends(current_user)):
    limit=max(1,min(limit,500)); offset=max(0,offset)
    selected_machine=machine(machine_id); conditions=["p.machine_id=%s"]; args:list=[selected_machine]
    if level:
        if level not in ("OK","WARN","CRITICAL"): raise HTTPException(400,"level must be OK, WARN, or CRITICAL")
        conditions.append("p.maintenance_level=%s"); args.append(level)
    where="WHERE "+" AND ".join(conditions)
    nm_hours=float(runtime_config.get("NEAR_MISS_TREND_WINDOW_HOURS",6))
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM spindle_predictions p {where}",args); total=cur.fetchone()["total"]
        cur.execute(f"""
          WITH nm AS (
            SELECT source.id, source.maintenance_level, trend.slope
            FROM spindle_predictions source
            LEFT JOIN LATERAL (
              SELECT regr_slope(sample.anomaly_score, extract(epoch from sample.tick_timestamp)) AS slope
              FROM spindle_predictions sample
              WHERE sample.machine_id=source.machine_id
                AND sample.tick_timestamp BETWEEN
                    source.tick_timestamp-(%s*interval '1 hour') AND source.tick_timestamp
            ) trend ON true
            WHERE source.machine_id=%s
          )
          SELECT p.*,
                 a.status AS alert_status, a.level AS alert_level, a.trigger AS alert_trigger,
                 a.reviewed_by AS alert_reviewed_by, a.reviewed_at AS alert_reviewed_at,
                 -- A prediction with no near_miss_reviews row hasn't necessarily
                 -- never been a near-miss — it may just not have been reviewed
                 -- yet. /api/near-miss treats "eligible, no row" as status
                 -- 'pending' (COALESCE(nmr.status,'pending')); this has to
                 -- recompute the same eligibility (maintenance_level='OK' AND
                 -- slope<0) or a pending near-miss silently disappears from
                 -- History instead of showing "Near miss - Pending" the way
                 -- the Near Miss queue itself does.
                 COALESCE(nmr.status,
                          CASE WHEN nm.maintenance_level='OK' AND nm.slope IS NOT NULL AND nm.slope<0
                               THEN 'pending' END) AS near_miss_status,
                 nmr.reviewed_by AS near_miss_reviewed_by, nmr.reviewed_at AS near_miss_reviewed_at
          FROM spindle_predictions p
          JOIN nm ON nm.id=p.id
          LEFT JOIN LATERAL (
            SELECT status, level, trigger, reviewed_by, reviewed_at FROM alerts a
            WHERE a.machine_id=p.machine_id AND a.tick_timestamp=p.tick_timestamp AND a.model_version=p.model_version
            ORDER BY a.id DESC LIMIT 1
          ) a ON true
          LEFT JOIN near_miss_reviews nmr ON nmr.prediction_id=p.id
          {where}
          ORDER BY p.tick_timestamp DESC LIMIT %s OFFSET %s""",[nm_hours,selected_machine]+args+[limit,offset]); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.get("/api/near-miss")
def near_miss(hours:float|None=None,limit:int=50,offset:int=0,status:str="pending",machine_id:str|None=None,user:User=Depends(current_user)):
    if status not in ("pending","acknowledged","flagged"): raise HTTPException(400,"status must be pending, acknowledged, or flagged")
    hours=float(hours if hours is not None else runtime_config.get("NEAR_MISS_TREND_WINDOW_HOURS",6))
    if not 0.25 <= hours <= 168: raise HTTPException(400,"hours must be between 0.25 and 168")
    limit=max(1,min(limit,500)); offset=max(0,offset)
    selected_machine=machine(machine_id)
    base_cte="""WITH x AS (
          SELECT source.id, source.machine_id, source.tick_timestamp, source.model_version,
                 source.anomaly_score, source.health_state, source.maintenance_level,
                 source.raw_reading, trend.slope
          FROM spindle_predictions source
          LEFT JOIN LATERAL (
            SELECT regr_slope(sample.anomaly_score, extract(epoch from sample.tick_timestamp)) AS slope
            FROM spindle_predictions sample
            WHERE sample.machine_id=source.machine_id
              AND sample.tick_timestamp BETWEEN
                  source.tick_timestamp-(%s*interval '1 hour') AND source.tick_timestamp
          ) trend ON true
          WHERE source.machine_id=%s)
          SELECT x.*, COALESCE(nmr.status,'pending') AS review_status, nmr.reviewed_by, nmr.reviewed_at
          FROM x LEFT JOIN near_miss_reviews nmr ON nmr.prediction_id=x.id
          WHERE x.maintenance_level='OK' AND x.slope IS NOT NULL AND x.slope < 0
            AND COALESCE(nmr.status,'pending')=%s"""
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM ({base_cte}) t",(hours,selected_machine,status)); total=cur.fetchone()["total"]
        cur.execute(base_cte+" ORDER BY x.health_state ASC LIMIT %s OFFSET %s",(hours,selected_machine,status,limit,offset)); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.post("/api/near-miss/{prediction_id}/review")
def review_near_miss(prediction_id:int,body:ReviewBody,user:User=Depends(current_user)):
    if body.decision not in ("acknowledged","flagged"): raise HTTPException(400,"decision must be acknowledged or flagged")
    c=conn()
    with c.cursor() as cur:
        cur.execute("SELECT id, machine_id, tick_timestamp, health_state FROM spindle_predictions WHERE id=%s",(prediction_id,)); row=cur.fetchone()
        if not row: c.rollback(); c.close(); raise HTTPException(404,"Near-miss record not found")
        cur.execute("""INSERT INTO near_miss_reviews(prediction_id,status,reviewed_by,reviewed_at)
                       VALUES(%s,%s,%s,now())
                       ON CONFLICT(prediction_id) DO UPDATE SET status=EXCLUDED.status,reviewed_by=EXCLUDED.reviewed_by,reviewed_at=now()""",
                    (prediction_id,body.decision,user.username))
        # A "flagged" near-miss is a human saying this OK-labeled tick looks
        # more like a missed detection than a healthy reading — i.e. a
        # suspected false negative. There was previously no path from that
        # judgment into the retraining/validation loop (Alert review's
        # "confirmed normal" already feeds reference_candidates; nothing
        # fed the opposite case). Reuse the existing regression_tests gate
        # (see retrain_service.run_shadow_retrain Gate 2) instead of adding
        # a new mechanism: any shadow model must keep scoring at least this
        # much risk at this exact flagged prediction timestamp, or promotion
        # is blocked. The surrounding range remains available as context only.
        regression_test_id=None
        if body.decision=="flagged":
            _, machine_id, tick_ts, health_state=row
            hours=float(runtime_config.get("NEAR_MISS_REGRESSION_WINDOW_HOURS",1))
            min_risk=runtime_config.regression_risk_floor(health_state)
            description=f"Near-miss prediction #{prediction_id}"
            cur.execute("SELECT id FROM regression_tests WHERE description=%s AND disabled_at IS NULL LIMIT 1",(description,))
            existing=cur.fetchone()
            if existing:
                regression_test_id=existing[0]
            else:
                cur.execute("""INSERT INTO regression_tests
                               (machine_id,description,timestamp_range,minimum_anomaly_risk,created_by,
                                target_prediction_id,target_timestamp)
                               VALUES(%s,%s,tstzrange(%s-(%s||' hours')::interval,%s+(%s||' hours')::interval,'[]'),%s,%s,%s,%s)
                               RETURNING id""",
                            (machine_id,description,tick_ts,hours,tick_ts,hours,min_risk,user.username,prediction_id,tick_ts))
                regression_test_id=cur.fetchone()[0]
    c.commit(); c.close(); return {"prediction_id":prediction_id,"status":body.decision,"regression_test_id":regression_test_id}

@app.websocket("/ws/live")
async def live(ws:WebSocket):
    try: session_user(ws.cookies.get(COOKIE))
    except HTTPException:
        await ws.close(code=4401); return
    try: selected_machine=machine(ws.query_params.get("machine_id"))
    except HTTPException:
        await ws.close(code=4400); return
    await ws.accept(); c=conn(); c.set_isolation_level(0)
    with c.cursor() as cur: cur.execute("LISTEN prediction_tick")
    try:
        while True:
            await asyncio.sleep(.5)
            if select.select([c],[],[],0)[0]:
                c.poll()
                while c.notifies:
                    note=c.notifies.pop(0)
                    payload=json.loads(note.payload)
                    if payload.get("machine_id")==selected_machine:
                        await ws.send_text(note.payload)
    except WebSocketDisconnect: pass
    finally: c.close()
