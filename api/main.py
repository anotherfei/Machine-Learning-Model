from __future__ import annotations
import asyncio
import json
import os
import re
import select
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import psycopg2.errors
import psycopg2.extras
from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from apscheduler.schedulers.background import BackgroundScheduler

import config
import artifact_utils
import db
import db_schema
import env_manager
import model_registry
import operating_state
import retrain_service
import retrain_jobs
import runtime_config
import backfill
import simulation_jobs
from api.auth import COOKIE, User, current_user, hash_password, issue_cookie, require_admin, session_user, verify_password
from api.contracts import (
    BackfillBody, EnvBody, LoginBody, ReviewBody, SimulationCaseBody,
    SimulationRunBody, ThresholdBody, TrainingConfigBody, UserCreate,
    ModelNameBody, OperatingStateOverrideBody,
)

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


_FRACTIONAL_TIMESTAMP = re.compile(
    r"^(.*[T ]\d{2}:\d{2}:\d{2})\.(\d+)(Z|[+-]\d{2}:\d{2})?$"
)
_LOCAL_TIMEZONE = datetime.now().astimezone().tzinfo or timezone.utc


def _comparison_timestamp(value) -> datetime | None:
    """Parse DB/state timestamps and normalize them to aware UTC datetimes."""
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        match = _FRACTIONAL_TIMESTAMP.match(text)
        if match:
            fraction = (match.group(2) + "000000")[:6]
            text = f"{match.group(1)}.{fraction}{match.group(3) or ''}"
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_LOCAL_TIMEZONE)
    return parsed.astimezone(timezone.utc)


def _register_current_artifacts(cur):
    """Register a newly trained root bundle and make it active.

    The manual trainer intentionally writes artifacts before the API starts.
    A fresh training metadata file has no version_id; that is the explicit
    signal that this bundle has not yet entered the database registry.
    """
    model_path=os.path.join(config.ARTIFACTS_DIR,"isolation_forest.pkl")
    metadata_path=os.path.join(config.ARTIFACTS_DIR,"metadata.json")
    if not os.path.exists(model_path) or not os.path.exists(metadata_path):
        # One-time compatibility migration from the former database-selected,
        # root-installed design. Once active.json exists, this query is never
        # used to choose the runtime model.
        if model_registry.active_version() is None:
            cur.execute(
                "SELECT version_id FROM model_versions WHERE status='active' "
                "ORDER BY promoted_at DESC NULLS LAST LIMIT 1"
            )
            row=cur.fetchone()
            if row:
                model_registry.activate_bundle(row[0])
        return False
    with open(metadata_path,encoding="utf-8") as handle:
        metadata=json.load(handle)

    version_id=metadata.get("version_id")
    if version_id:
        cur.execute("SELECT status FROM model_versions WHERE version_id=%s",(version_id,))
        registered=cur.fetchone()
        if registered:
            if registered[0] == "active":
                model_registry.activate_bundle(version_id)
            elif model_registry.active_version() is None:
                cur.execute(
                    "SELECT version_id FROM model_versions WHERE status='active' "
                    "ORDER BY promoted_at DESC NULLS LAST LIMIT 1"
                )
                active_row=cur.fetchone()
                if active_row:
                    model_registry.activate_bundle(active_row[0])
            return True
        path=model_registry.bundle_path(version_id)
        if not os.path.isdir(path):
            path=model_registry.snapshot_current(version_id)
    else:
        version_id=model_registry.new_version_id()
        path=model_registry.snapshot_current(version_id)

    # The snapshot is complete before its small active pointer is changed.
    # Load registration evidence from that exact immutable directory.
    model_registry.activate_bundle(version_id)
    reference_rows=artifact_utils.load_reference_rows(path)
    if reference_rows is not None:
        signature_source=[f"{row['machine_id']}\0{row['timestamp']}" for row in reference_rows]
    else:
        timestamps=artifact_utils.load_reference_timestamps(path)
        signature_source=timestamps if timestamps is not None else []
    signature=model_registry.reference_signature(signature_source)

    cur.execute("UPDATE model_versions SET status='retired' WHERE status='active'")
    cur.execute(
        """INSERT INTO model_versions
           (version_id,artifact_path,reference_signature,status,promoted_at,promoted_by)
           VALUES(%s,%s,%s,'active',now(),'manual-training-bootstrap')""",
        (version_id,path,signature),
    )
    return True


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
        clear_staging=_register_current_artifacts(cur)
    c.commit(); c.close()
    if clear_staging:
        # The committed immutable version and active pointer are now the only
        # production copy. Root files were merely manual-training staging.
        try:
            model_registry.clear_root_staging()
        except OSError as exc:
            # Registration is already durable and readers use active.json, so
            # stale staging files are harmless and can be cleaned next start.
            print(f"[model-registry] Could not remove root staging files: {exc}")


def _scheduled_retrain():
    c=None
    try:
        c=conn()
        queued_ids=retrain_jobs.queued_ids(c)
        if queued_ids:
            for job_id in queued_ids: _schedule_retrain_job(job_id)
            return
        if not runtime_config.get("AUTO_RETRAIN_ENABLED"):
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
    minutes=int(runtime_config.get("RETRAIN_CHECK_INTERVAL_MINUTES"))
    scheduler.add_job(
        _scheduled_retrain,"interval",minutes=minutes,
        id="retrain-check",replace_existing=True,coalesce=True,max_instances=1,
    )


def _schedule_retrain_job(job_id:int):
    scheduler.add_job(
        retrain_jobs.execute,"date",run_date=datetime.now(timezone.utc),args=[job_id],
        id=f"retrain-job-{job_id}",replace_existing=True,misfire_grace_time=3600,
    )

def _schedule_simulation_job(run_id:int):
    scheduler.add_job(
        simulation_jobs.execute,"date",run_date=datetime.now(timezone.utc),args=[run_id],
        id=f"simulation-run-{run_id}",replace_existing=True,misfire_grace_time=3600,
    )

@app.on_event("startup")
def startup():
    if env_manager.ensure_secure_app_secret():
        print("[security] Replaced an absent or placeholder APP_SECRET_KEY; existing sessions are invalid")
    _bootstrap()
    _configure_retrain_job()
    c=conn()
    try:
        with c.cursor() as cur:
            cur.execute(
                """UPDATE state
                   SET value=value || jsonb_build_object(
                         'status','failed',
                         'error','The previous application session ended before backfill completed.',
                         'finished_at',now()
                       ),
                       updated_at=now(),updated_by='api-startup'
                   WHERE namespace='backfill' AND key='startup'
                     AND value->>'status' IN ('preparing','running','restarting','draining')"""
            )
        c.commit()
        queued_jobs=retrain_jobs.recover_interrupted(c)
        queued_simulations=simulation_jobs.recover_interrupted(c)
    finally: c.close()
    runtime_status=backfill.read_runtime_status()
    if runtime_status and runtime_status.get("status") in {
        "launching","connecting","preparing","running","restarting","draining",
    }:
        backfill.write_runtime_status({
            **runtime_status,
            "status":"failed",
            "error":"The previous application session ended before backfill completed.",
            "finished_at":datetime.now(timezone.utc).isoformat(),
            "updated_at":datetime.now(timezone.utc).isoformat(),
        })
    for job_id in queued_jobs: _schedule_retrain_job(job_id)
    for run_id in queued_simulations: _schedule_simulation_job(run_id)
    scheduler.start()


@app.on_event("shutdown")
def shutdown():
    if scheduler.running:
        scheduler.shutdown(wait=False)

@app.get("/api/health")
def health():
    return {"ok": True, "mode": "production"}


@app.get("/api/backfill/status")
def backfill_status(user:User=Depends(current_user)):
    return backfill.read_runtime_status() or {"status":"idle","progress":0.0}


def _catchup_is_active(c) -> bool:
    runtime_status=backfill.read_runtime_status()
    if runtime_status is not None:
        return runtime_status.get("status") in {
            "launching","connecting","preparing","running","restarting","draining",
        }
    with c.cursor() as cur:
        cur.execute(
            """SELECT COALESCE(value->>'status','') IN ('preparing','running','restarting','draining')
               FROM state WHERE namespace='backfill' AND key='startup'"""
        )
        row=cur.fetchone()
    return bool(row and row[0])


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
    if not 1 <= days <= 7:
        raise HTTPException(400,"days must be between 1 and 7")
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
    display_start=start
    display_end=end
    if rows:
        first_bucket=min(row["bucket"] for row in rows)
        last_bucket=max(row["bucket"] for row in rows)
        if first_bucket == last_bucket:
            display_start=max(start,first_bucket-timedelta(minutes=30))
            display_end=min(end,last_bucket+timedelta(minutes=30))
        else:
            display_start=first_bucket
            display_end=last_bucket
    coverage_seconds=max(0.0,(display_end-display_start).total_seconds())
    return {
        "days":coverage_seconds/86400.0,
        "requested_days":days,
        "coverage_seconds":coverage_seconds,
        "metric":"hourly_average_condition",
        "start":display_start,"end":display_end,
        "series":[{"machine_id":machine_id,"points":grouped.get(machine_id,[])} for machine_id in machine_ids],
    }


def _latest_live_response(c,selected_machine:str):
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
                "SELECT value FROM state WHERE namespace='machine' AND key=%s",
                (selected_machine,),
            )
            runtime_row=cur.fetchone()
            runtime_state=runtime_row["value"] if runtime_row else None
        operator_override=operating_state.load_override(c,selected_machine)
    except HTTPException:
        raise
    except Exception as exc:
        c.rollback()
        raise HTTPException(503,f"Cannot read live PostgreSQL source for {selected_machine}: {exc}")

    source_timestamp=source_row[columns["timestamp"]]
    prediction_timestamp=prediction["tick_timestamp"] if prediction is not None else None
    source_comparison_timestamp=_comparison_timestamp(source_timestamp)
    prediction_comparison_timestamp=_comparison_timestamp(prediction_timestamp)
    source_age_seconds=None
    if source_comparison_timestamp is not None:
        source_age_seconds=max(
            0.0,
            (datetime.now(timezone.utc)-source_comparison_timestamp).total_seconds(),
        )

    operating="UNKNOWN"
    operating_reason="The worker has not published an operating state yet."
    operating_confidence=0.0
    state_changed_at=None
    runtime_tick=None
    if runtime_state:
        detected_state={
            "state":runtime_state.get("detected_operating_state",runtime_state["operating_state"]),
            "reason":runtime_state.get("detected_reason",runtime_state["reason"]),
            "confidence":runtime_state.get("detected_confidence",runtime_state["confidence"]),
            "activity_score":runtime_state.get("activity_score"),
            "low_motion":runtime_state.get("low_motion",False),
            "stop_threshold":runtime_state.get("stop_threshold"),
            "run_threshold":runtime_state.get("run_threshold"),
            "changed":False,
        }
        effective_state=operating_state.apply_operator_override(
            detected_state,operator_override,datetime.now(timezone.utc)
        )
        operating=effective_state["state"]
        operating_reason=effective_state["reason"]
        operating_confidence=effective_state["confidence"]
        state_changed_at=runtime_state["state_changed_at"]
        runtime_tick=runtime_state["tick_timestamp"]
        state_changed_at=_comparison_timestamp(state_changed_at)
        runtime_tick=_comparison_timestamp(runtime_tick)
    else:
        effective_state=operating_state.apply_operator_override({
            "state":operating,"reason":operating_reason,"confidence":operating_confidence,
            "activity_score":None,"low_motion":False,"stop_threshold":None,
            "run_threshold":None,"changed":False,
        },operator_override,datetime.now(timezone.utc))
        operating=effective_state["state"]
        operating_reason=effective_state["reason"]
        operating_confidence=effective_state["confidence"]
    if effective_state["state_source"] == "operator" and operator_override:
        state_changed_at=_comparison_timestamp(operator_override.get("set_at"))
    stale_seconds=runtime_config.get("SOURCE_STALE_SECONDS",config.SOURCE_STALE_SECONDS)
    if source_age_seconds is not None and source_age_seconds > stale_seconds:
        operating="NO_DATA"
        operating_reason=f"Newest source row is {int(source_age_seconds)} seconds old; waiting for fresh sensor data."
        operating_confidence=1.0
        effective_state["state_source"]="source_guard"

    prediction_lag_seconds=None
    prediction_matches_source=False
    if source_comparison_timestamp is not None and prediction_comparison_timestamp is not None:
        prediction_matches_source=prediction_comparison_timestamp == source_comparison_timestamp
        prediction_lag_seconds=max(
            0.0,
            (source_comparison_timestamp-prediction_comparison_timestamp).total_seconds(),
        )
    allowed_prediction_lag_seconds=max(
        float(stale_seconds),
        3.0*float(runtime_config.get("WORKER_POLL_SECONDS",config.WORKER_POLL_SECONDS)),
    )
    prediction_is_current=(
        prediction is not None
        and operating in ("RUNNING","UNKNOWN")
        and prediction_lag_seconds is not None
        and prediction_lag_seconds <= allowed_prediction_lag_seconds
    )
    if prediction_is_current and prediction_comparison_timestamp is not None and state_changed_at is not None:
        prediction_is_current=prediction_comparison_timestamp >= state_changed_at
    if prediction_is_current and prediction_comparison_timestamp is not None and runtime_tick is not None:
        prediction_is_current=prediction_comparison_timestamp >= runtime_tick

    response={
        "machine_id":selected_machine,
        "timestamp":source_timestamp,
        "source":"postgresql",
        "source_table":db.get_table_name(),
        "source_age_seconds":source_age_seconds,
        "prediction_lag_seconds":prediction_lag_seconds,
        "prediction_matches_source":prediction_matches_source,
        "prediction_delayed":bool(prediction_is_current and not prediction_matches_source),
        "allowed_prediction_lag_seconds":allowed_prediction_lag_seconds,
        "worker_poll_seconds":runtime_config.get("WORKER_POLL_SECONDS",config.WORKER_POLL_SECONDS),
        "prediction_timestamp":prediction_timestamp,
        "prediction_available":prediction_is_current,
        "operating_state":operating,
        "operating_state_reason":operating_reason,
        "operating_state_confidence":operating_confidence,
        "operating_state_source":effective_state["state_source"],
        "detected_operating_state":effective_state["detected_state"],
        "detected_operating_state_reason":effective_state["detected_reason"],
        "operating_override":operator_override,
        "operating_state_changed_at":state_changed_at,
        "operating_state_activity":runtime_state["activity_score"] if runtime_state else None,
        "operating_state_stop_threshold":runtime_state["stop_threshold"] if runtime_state else None,
        "operating_state_run_threshold":runtime_state["run_threshold"] if runtime_state else None,
        **raw,
    }
    if prediction is not None and not prediction_is_current and operating in ("RUNNING","UNKNOWN"):
        lag_text=(
            f" The latest prediction is {prediction_lag_seconds:.1f} seconds behind."
            if prediction_lag_seconds is not None else ""
        )
        response["prediction_wait_reason"]=(
            "The latest coherent inference result is older than the permitted freshness window. "
            "Condition and maintenance values are hidden until the worker catches up."
            f"{lag_text} The worker's caught-up polling interval is "
            f"{runtime_config.get('WORKER_POLL_SECONDS',config.WORKER_POLL_SECONDS)} second(s)."
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


@app.get("/api/live/latest")
def latest_live(machine_id:str|None=None,user:User=Depends(current_user)):
    c=conn()
    try:
        return _latest_live_response(c,machine(machine_id))
    finally:
        c.close()


@app.get("/api/fleet/latest")
def fleet_latest(machine_ids:str,user:User=Depends(current_user)):
    requested=[]
    for value in machine_ids.split(","):
        if not value.strip():
            continue
        selected=machine(value)
        if selected not in requested:
            requested.append(selected)
    if not requested:
        raise HTTPException(400,"machine_ids must contain at least one machine")
    if len(requested)>100:
        raise HTTPException(400,"machine_ids supports at most 100 machines")
    c=conn();items=[];unavailable=[]
    try:
        # Run source lookups sequentially on one connection. This avoids the
        # browser creating a concurrent query storm while historical catch-up
        # is already reading a large production table.
        for selected in requested:
            try:
                items.append(_latest_live_response(c,selected))
            except HTTPException as exc:
                unavailable.append({"machine_id":selected,"detail":exc.detail})
        return {"items":items,"unavailable":unavailable}
    finally:
        c.close()


@app.post("/api/machines/{machine_id}/operating-override")
def set_operating_override(
    machine_id:str,body:OperatingStateOverrideBody,admin:User=Depends(require_admin)
):
    selected=machine(machine_id)
    c=conn()
    try:
        if db.fetch_latest_row(c,db.get_table_name(),selected) is None:
            raise HTTPException(404,f"No source rows found for machine {selected}")
        try:
            document=operating_state.save_override(
                c,selected,body.state,body.expires_minutes,admin.username,body.note
            )
        except ValueError as exc:
            raise HTTPException(400,str(exc))
        c.commit()
        return {"machine_id":selected,"override":document}
    except HTTPException:
        c.rollback()
        raise
    finally:
        c.close()


@app.delete("/api/machines/{machine_id}/operating-override")
def delete_operating_override(machine_id:str,admin:User=Depends(require_admin)):
    selected=machine(machine_id)
    c=conn()
    try:
        removed=operating_state.clear_override(c,selected)
        c.commit()
        return {"machine_id":selected,"cleared":removed,"cleared_by":admin.username}
    finally:
        c.close()

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
    c=conn()
    try:
        if _catchup_is_active(c):
            raise HTTPException(409,"Environment changes are locked until sequential catch-up finishes.")
    finally:
        c.close()
    try: changed=env_manager.write(body.values)
    except ValueError as e: raise HTTPException(400,str(e))
    c=conn()
    with c.cursor() as cur:
        cur.execute(
            """INSERT INTO state(namespace,key,value,updated_by)
               VALUES(
                 'env_audit',
                 concat(extract(epoch from clock_timestamp())::numeric::text,'-',txid_current()::text),
                 jsonb_build_object('changed_keys',%s::jsonb),%s
               )""",
            (json.dumps(changed),admin.username),
        )
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
    c=conn()
    try:
        if _catchup_is_active(c):
            raise HTTPException(409,"Runtime policy changes are locked until sequential catch-up finishes.")
        runtime_config.save_to_db(c,values,admin.username)
    finally:
        c.close()
    return values

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
    where=["alert_status=%s","machine_id=%s"]; args=[status,machine(machine_id)]
    if status=="pending":
        active_version=model_registry.active_version()
        if active_version:
            where.append("model_version=%s"); args.append(active_version)
    if trigger: where.append("maintenance_trigger=%s"); args.append(trigger)
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM spindle_predictions WHERE {' AND '.join(where)}",args); total=cur.fetchone()["total"]
        cur.execute(f"""SELECT id,machine_id,tick_timestamp,model_version,
                               maintenance_trigger AS trigger,maintenance_level AS level,
                               health_state,anomaly_score,raw_reading,feature_vector,
                               alert_status AS status,alert_reviewed_by AS reviewed_by,
                               alert_reviewed_at AS reviewed_at,created_at
                        FROM spindle_predictions WHERE {' AND '.join(where)}
                        ORDER BY tick_timestamp DESC LIMIT %s OFFSET %s""",args+[limit,offset]); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.post("/api/alerts/{alert_id}/review")
def review(alert_id:int,body:ReviewBody,user:User=Depends(current_user)):
    if body.decision not in ("confirmed_anomaly","confirmed_normal"): raise HTTPException(400,"decision must be confirmed_anomaly or confirmed_normal")
    c=conn()
    with c.cursor() as cur:
        cur.execute(
            """UPDATE spindle_predictions
               SET alert_status=%s,alert_reviewed_by=%s,alert_reviewed_at=now(),
                   reference_candidate_at=CASE WHEN %s='confirmed_normal'
                     THEN COALESCE(reference_candidate_at,now()) ELSE reference_candidate_at END
               WHERE id=%s AND alert_status='pending' RETURNING tick_timestamp""",
            (body.decision,user.username,body.decision,alert_id),
        ); row=cur.fetchone()
        if not row: c.rollback(); c.close(); raise HTTPException(409,"Alert not found or already reviewed")
    c.commit(); c.close(); return {"id":alert_id,"status":body.decision}


@app.get("/api/alerts/{alert_id}/context")
def alert_context(alert_id:int,hours:float=3,user:User=Depends(current_user)):
    if not 0.25 <= hours <= 168: raise HTTPException(400,"hours must be between 0.25 and 168")
    c=conn()
    with c.cursor() as cur:
        cur.execute("SELECT tick_timestamp,machine_id FROM spindle_predictions WHERE id=%s AND alert_status IS NOT NULL",(alert_id,)); row=cur.fetchone()
        if not row: c.close(); raise HTTPException(404,"Alert not found")
        ts,machine_id=row
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT tick_timestamp,health_state,anomaly_score,maintenance_level FROM spindle_predictions
                       WHERE machine_id=%s AND tick_timestamp BETWEEN %s-(%s||' hours')::interval AND %s+(%s||' hours')::interval
                       ORDER BY tick_timestamp""",(machine_id,ts,hours,ts,hours)); rows=cur.fetchall()
    c.close(); return rows

@app.get("/api/simulations")
def simulations(limit:int=25,user:User=Depends(current_user)):
    return simulation_jobs.list_runs(limit)

@app.get("/api/simulation-templates")
def simulation_templates(limit:int=50,user:User=Depends(current_user)):
    return simulation_jobs.list_runs(limit,drafts=True)

def _simulation_template_values(body:SimulationRunBody):
    name=body.name.strip()
    if not name or len(name)>200:
        raise HTTPException(400,"name must contain 1 to 200 characters")
    if not 1<=len(body.cases)<=simulation_jobs.MAX_CASES:
        raise HTTPException(400,f"cases must contain 1 to {simulation_jobs.MAX_CASES} labelled event ranges")
    cases=[]
    for index,item in enumerate(body.cases,1):
        expected=item.expected_status.strip().upper()
        if expected not in simulation_jobs.EXPECTED_STATUSES:
            raise HTTPException(400,f"case {index}: unsupported expected_status {expected!r}")
        description=item.description.strip()
        if len(description)>500:
            raise HTTPException(400,f"case {index}: description cannot exceed 500 characters")
        cases.append({
            "machine_id":machine(item.machine_id),
            "description":description,
            "start":item.start.strip(),
            "end":item.end.strip(),
            "expected_status":expected,
        })
    return name,cases

@app.post("/api/simulation-templates")
def create_simulation_template(body:SimulationRunBody,admin:User=Depends(require_admin)):
    name,cases=_simulation_template_values(body)
    try:
        return simulation_jobs.save_draft(name,cases,admin.username)
    except ValueError as exc:
        raise HTTPException(400,str(exc))

@app.put("/api/simulation-templates/{template_id}")
def update_simulation_template(template_id:int,body:SimulationRunBody,admin:User=Depends(require_admin)):
    name,cases=_simulation_template_values(body)
    try:
        return simulation_jobs.save_draft(name,cases,admin.username,template_id)
    except ValueError as exc:
        raise HTTPException(404,str(exc))

@app.put("/api/simulation-templates/{template_id}/validation-suite")
def set_simulation_validation_suite(template_id:int,enabled:bool=True,admin:User=Depends(require_admin)):
    try:
        return simulation_jobs.set_validation_suite(template_id,enabled)
    except ValueError as exc:
        raise HTTPException(404,str(exc))

@app.delete("/api/simulation-templates/{template_id}")
def delete_simulation_template(template_id:int,admin:User=Depends(require_admin)):
    if not simulation_jobs.delete_run(template_id):
        raise HTTPException(404,"Reusable simulation list not found")
    return {"deleted":template_id}

@app.get("/api/simulations/{run_id}")
def simulation(run_id:int,user:User=Depends(current_user)):
    item=simulation_jobs.get_run(run_id)
    if not item: raise HTTPException(404,"Simulation run not found")
    return item

@app.post("/api/simulations",status_code=202)
def create_simulation(body:SimulationRunBody,admin:User=Depends(require_admin)):
    name=body.name.strip()
    if not name or len(name)>200: raise HTTPException(400,"name must contain 1 to 200 characters")
    if not 1<=len(body.cases)<=simulation_jobs.MAX_CASES:
        raise HTTPException(400,f"cases must contain 1 to {simulation_jobs.MAX_CASES} labelled event ranges")
    parsed=[]
    for index,item in enumerate(body.cases,1):
        description=item.description.strip()
        if not description or len(description)>500:
            raise HTTPException(400,f"case {index}: description must contain 1 to 500 characters")
        expected=item.expected_status.strip().upper()
        if expected not in simulation_jobs.EXPECTED_STATUSES:
            raise HTTPException(400,f"case {index}: unsupported expected_status {expected!r}")
        try:
            start=datetime.fromisoformat(item.start.replace("Z","+00:00"))
            end=datetime.fromisoformat(item.end.replace("Z","+00:00"))
        except ValueError: raise HTTPException(400,f"case {index}: start and end must be ISO-8601 timestamps")
        if start.tzinfo is None or end.tzinfo is None:
            raise HTTPException(400,f"case {index}: start and end must include a timezone")
        if end<=start:
            raise HTTPException(400,f"case {index}: end must be later than start")
        if end-start>timedelta(days=simulation_jobs.MAX_EVENT_WINDOW_DAYS):
            raise HTTPException(400,f"case {index}: event range cannot exceed {simulation_jobs.MAX_EVENT_WINDOW_DAYS} days")
        if end>datetime.now(timezone.utc):
            raise HTTPException(400,f"case {index}: event end cannot be in the future")
        parsed.append({
            "machine_id":machine(item.machine_id),"description":description,
            "start":start,"end":end,"expected_status":expected,
        })
    try:
        requested_version=(body.model_version or "").strip()
        version_id=requested_version or model_registry.active_version()
        if not version_id: raise HTTPException(409,"No active model is available for simulation")
        c=conn()
        try:
            with c.cursor() as cur:
                cur.execute("SELECT 1 FROM model_versions WHERE version_id=%s",(version_id,))
                if cur.fetchone() is None:
                    raise HTTPException(404,f"Model version {version_id!r} is not registered")
        finally:
            c.close()
        commissioned=model_registry.commissioned_machines(version_id)
        for item in parsed:
            machine_id=item["machine_id"]
            if machine_id not in commissioned:
                raise HTTPException(400,f"Machine {machine_id!r} is not commissioned in selected model {version_id!r}")
        run=simulation_jobs.enqueue(name,version_id,parsed,admin.username)
        run_id=int(run["id"])
    except ValueError as exc:
        raise HTTPException(409,str(exc))
    _schedule_simulation_job(int(run_id))
    return {"id":run_id,"status":"queued","model_version":version_id}

@app.delete("/api/simulations/{run_id}")
def delete_simulation(run_id:int,admin:User=Depends(require_admin)):
    if not simulation_jobs.delete_run(run_id):
        raise HTTPException(404,"Reusable list or completed simulation run not found")
    return {"deleted":run_id}

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
        cur.execute(
            """SELECT version_id,display_name,artifact_path,reference_signature,status,
                      validation_report,created_at,promoted_at,promoted_by
               FROM model_versions ORDER BY created_at DESC"""
        ); rows=[dict(row) for row in cur.fetchall()]
    c.close()
    active_id=model_registry.active_version()
    for row in rows:
        commissioned=sorted(model_registry.commissioned_machines(row["version_id"]))
        row["commissioned_machines"]=len(commissioned)
        row["commissioned_machine_ids"]=commissioned
        if row["version_id"] == active_id:
            row["status"]="active"
        elif row["status"] == "active":
            row["status"]="retired"
    return rows
@app.patch("/api/models/{version_id}/name")
def rename_model(version_id:str,body:ModelNameBody,admin:User=Depends(require_admin)):
    c=conn()
    try: name=model_registry.rename_version(c,version_id,body.name)
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()
    return {"version_id":version_id,"display_name":name}
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
    nm_hours=float(runtime_config.get("NEAR_MISS_TREND_WINDOW_HOURS"))
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
                 CASE WHEN p.alert_status IS NOT NULL THEN p.maintenance_level END AS alert_level,
                 CASE WHEN p.alert_status IS NOT NULL THEN p.maintenance_trigger END AS alert_trigger,
                 -- A prediction with no saved near-miss decision hasn't necessarily
                 -- never been a near-miss — it may just not have been reviewed
                 -- yet. /api/near-miss treats "eligible, no row" as status
                 -- 'pending'; this has to
                 -- recompute the same eligibility (maintenance_level='OK' AND
                 -- slope<0) or a pending near-miss silently disappears from
                 -- History instead of showing "Near miss - Pending" the way
                 -- the Near Miss queue itself does.
                 COALESCE(p.near_miss_status,
                          CASE WHEN nm.maintenance_level='OK' AND nm.slope IS NOT NULL AND nm.slope<0
                               THEN 'pending' END) AS effective_near_miss_status
          FROM spindle_predictions p
          JOIN nm ON nm.id=p.id
          {where}
          ORDER BY p.tick_timestamp DESC LIMIT %s OFFSET %s""",[nm_hours,selected_machine]+args+[limit,offset])
        rows=[dict(row) for row in cur.fetchall()]
        for row in rows:
            row["near_miss_status"]=row.pop("effective_near_miss_status")
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.get("/api/near-miss")
def near_miss(hours:float|None=None,limit:int=50,offset:int=0,status:str="pending",machine_id:str|None=None,user:User=Depends(current_user)):
    if status not in ("pending","acknowledged","flagged"): raise HTTPException(400,"status must be pending, acknowledged, or flagged")
    hours=float(hours if hours is not None else runtime_config.get("NEAR_MISS_TREND_WINDOW_HOURS"))
    if not 0.25 <= hours <= 168: raise HTTPException(400,"hours must be between 0.25 and 168")
    limit=max(1,min(limit,500)); offset=max(0,offset)
    selected_machine=machine(machine_id)
    base_cte="""WITH x AS (
          SELECT source.id, source.machine_id, source.tick_timestamp, source.model_version,
                 source.anomaly_score, source.health_state, source.maintenance_level,
                 source.raw_reading,source.near_miss_status,
                 source.near_miss_reviewed_by,source.near_miss_reviewed_at,trend.slope
          FROM spindle_predictions source
          LEFT JOIN LATERAL (
            SELECT regr_slope(sample.anomaly_score, extract(epoch from sample.tick_timestamp)) AS slope
            FROM spindle_predictions sample
            WHERE sample.machine_id=source.machine_id
              AND sample.tick_timestamp BETWEEN
                  source.tick_timestamp-(%s*interval '1 hour') AND source.tick_timestamp
          ) trend ON true
          WHERE source.machine_id=%s)
          SELECT x.*, COALESCE(x.near_miss_status,'pending') AS review_status,
                 x.near_miss_reviewed_by AS reviewed_by,
                 x.near_miss_reviewed_at AS reviewed_at
          FROM x
          WHERE x.maintenance_level='OK' AND x.slope IS NOT NULL AND x.slope < 0
            AND COALESCE(x.near_miss_status,'pending')=%s"""
    if status=="pending":
        active_version=model_registry.active_version()
        if active_version:
            base_cte += " AND x.model_version=%s"
    c=conn()
    query_args=[hours,selected_machine,status]+([active_version] if status=="pending" and active_version else [])
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM ({base_cte}) t",query_args); total=cur.fetchone()["total"]
        cur.execute(base_cte+" ORDER BY x.health_state ASC LIMIT %s OFFSET %s",query_args+[limit,offset]); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.post("/api/near-miss/{prediction_id}/review")
def review_near_miss(prediction_id:int,body:ReviewBody,user:User=Depends(current_user)):
    if body.decision not in ("acknowledged","flagged"): raise HTTPException(400,"decision must be acknowledged or flagged")
    c=conn()
    with c.cursor() as cur:
        cur.execute(
            """UPDATE spindle_predictions
               SET near_miss_status=%s,near_miss_reviewed_by=%s,near_miss_reviewed_at=now()
               WHERE id=%s RETURNING id""",
            (body.decision,user.username,prediction_id),
        ); row=cur.fetchone()
        if not row: c.rollback(); c.close(); raise HTTPException(404,"Near-miss record not found")
        # Flagging remains an auditable human review. Historical labelled
        # evaluation now belongs to the explicit simulation workspace.
    c.commit(); c.close(); return {"prediction_id":prediction_id,"status":body.decision}

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
