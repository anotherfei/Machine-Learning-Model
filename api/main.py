from __future__ import annotations
import asyncio
import json
import os
import select
import time
from collections import defaultdict, deque

import psycopg2.extras
from fastapi import Depends, FastAPI, HTTPException, Response, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from apscheduler.schedulers.background import BackgroundScheduler

import config
import db
import db_schema
import env_manager
import model_registry
import recalibrate_service
import retrain_service
import runtime_config
import backfill
from api.auth import COOKIE, User, current_user, hash_password, issue_cookie, require_admin, verify_password

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
                cur.execute("INSERT INTO app_users(username,password_hash,role) VALUES(%s,%s,'admin')", (username,hash_password(password)))
        # Register legacy/current bundle once if artifacts exist and registry is empty.
        cur.execute("SELECT count(*) FROM model_versions"); mv=cur.fetchone()[0]
        if mv==0 and os.path.exists(os.path.join(os.path.dirname(__file__),"..","artifacts","isolation_forest.pkl")):
            vid=model_registry.new_version_id(); path=model_registry.snapshot_current(vid)
            # snapshot_current stamps the version ID into the versioned copy.
            # Install that copy back to the active artifact path before the
            # worker starts so predictions and per-machine calibration lookups
            # use the same registry version ID from the first tick onward.
            model_registry.install_bundle(vid)
            import artifact_utils
            reference_rows=artifact_utils.load_reference_rows()
            if reference_rows is not None:
                signature_source=[f"{row['machine_id']}\0{row['timestamp']}" for row in reference_rows]
            else:
                ts=artifact_utils.load_reference_timestamps(); signature_source=ts if ts is not None else []
            signature=model_registry.reference_signature(signature_source)
            cur.execute("INSERT INTO model_versions(version_id,artifact_path,reference_signature,status,promoted_at,promoted_by) VALUES(%s,%s,%s,'active',now(),'bootstrap')",(vid,path,signature))
            for machine_id, item in (artifact_utils.load_machine_calibrations() or {}).items():
                cur.execute(
                    """INSERT INTO model_calibrations(version_id,machine_id,calibration,source_rows,source_description,created_by)
                       VALUES(%s,%s,%s::jsonb,%s,%s,'bootstrap') RETURNING id""",
                    (vid,machine_id,json.dumps(item["calibration"]),item["source_rows"],
                     "Initial calibration from balanced pooled commissioning training"),
                )
                calibration_id=cur.fetchone()[0]
                cur.execute(
                    """INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id)
                       VALUES(%s,%s,%s)""",
                    (machine_id,vid,calibration_id),
                )
    c.commit(); c.close()


def _scheduled_retrain():
    if not runtime_config.get("AUTO_RETRAIN_ENABLED", True):
        return
    try:
        c=conn(); retrain_service.run_shadow_retrain(c, force=False); c.close()
    except Exception as e:
        print(f"[retrain-job] {e}")

@app.on_event("startup")
def startup():
    _bootstrap()
    scheduler.add_job(_scheduled_retrain,"interval",hours=1,id="retrain-check",replace_existing=True)
    scheduler.start()

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
    RETRAIN_BATCH_SIZE: float
    RETRAIN_TIME_CAP_DAYS: float
    REFERENCE_WINDOW_MONTHS: float
    REFERENCE_DEDUP_WINDOW_HOURS: float
    REFERENCE_COSINE_SIMILARITY: float
    AUTO_RETRAIN_ENABLED: bool = True
class RecalibrateBody(BaseModel): hours: float = 24; min_rows: int = 200; source: str = "spec_bounds"; machine_id: str = db.DEFAULT_MACHINE_ID
class CalibrationActivateBody(BaseModel): calibration_id: int | None = None

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
    with c.cursor() as cur:
        cur.execute("""SELECT machine_id FROM (
                         SELECT DISTINCT machine_id FROM spindle_predictions
                         UNION SELECT DISTINCT machine_id FROM alerts
                       ) machines ORDER BY machine_id""")
        items={row[0] for row in cur.fetchall() if row[0]}
    try:
        items.update(db.fetch_machine_ids(c,db.get_table_name()))
    except Exception as exc:
        # Persisted predictions still provide a usable selector if the raw
        # source table is temporarily unavailable. The worker will report
        # the underlying source error through its normal operational path.
        c.rollback()
        print(f"[machines] Source-table discovery unavailable: {exc}")
    c.close()
    items=sorted(items)
    if not items: items=[db.DEFAULT_MACHINE_ID]
    return {"items":items,"default":db.DEFAULT_MACHINE_ID if db.DEFAULT_MACHINE_ID in items else items[0]}

@app.post("/api/users")
def create_user(body:UserCreate,admin:User=Depends(require_admin)):
    if body.role not in ("viewer","admin"): raise HTTPException(400,"role must be viewer or admin")
    c=conn()
    try:
        with c.cursor() as cur: cur.execute("INSERT INTO app_users(username,password_hash,role) VALUES(%s,%s,%s)",(body.username,hash_password(body.password),body.role))
        c.commit()
    except Exception as e: c.rollback(); raise HTTPException(400,str(e))
    finally: c.close()
    return {"username":body.username,"role":body.role}

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
    return {k:runtime_config.get(k) for k in ("MAINTENANCE_PROB_URGENT","MAINTENANCE_PROB_PLAN","FAILURE_HEALTH_THRESHOLD","MAINTENANCE_HEALTH_INSPECT")}
@app.put("/api/config/thresholds")
def set_thresholds(body:ThresholdBody,admin:User=Depends(require_admin)):
    values=body.model_dump()
    try: runtime_config.validate_thresholds(values)
    except ValueError as e: raise HTTPException(400,str(e))
    c=conn(); runtime_config.save_to_db(c,values,admin.username); c.close(); return values

@app.get("/api/config/spec-bounds")
def spec_bounds(user:User=Depends(current_user)):
    # config.SPEC_MAX is a source-level constant (see config.py) — unlike
    # thresholds/training it isn't runtime_config-backed, so there's no PUT:
    # changing it means editing config.py and redeploying, not a form here.
    return {"bounds":config.SPEC_MAX,"editable":False}

@app.get("/api/config/training")
def training_config(user:User=Depends(current_user)):
    # Same knobs retrain_service.py already reads via runtime_config.get()
    # (should_retrain(), _dedup(), _current_reference_features()) — this
    # just surfaces them for the Models page instead of requiring a direct
    # DB write to change what "training" does. AUTO_RETRAIN_ENABLED gates
    # _scheduled_retrain() (the hourly APScheduler job) directly; it
    # doesn't affect the "Run shadow retrain" button, which is always a
    # deliberate manual action regardless of this setting.
    keys=(*runtime_config.TRAINING_KEYS,"AUTO_RETRAIN_ENABLED")
    return {k:runtime_config.get(k) for k in keys}
@app.put("/api/config/training")
def set_training_config(body:TrainingConfigBody,admin:User=Depends(require_admin)):
    values=body.model_dump()
    try: runtime_config.validate_training_config(values)
    except ValueError as e: raise HTTPException(400,str(e))
    c=conn(); runtime_config.save_to_db(c,values,admin.username); c.close(); return values

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
def alert_context(alert_id:int,hours:int=3,user:User=Depends(current_user)):
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
        cur.execute("SELECT * FROM regression_tests ORDER BY created_at DESC"); rows=cur.fetchall()
    c.close(); return rows

@app.post("/api/regression-tests")
def add_regression(body:RegressionBody,admin:User=Depends(require_admin)):
    if not 0<=body.minimum_anomaly_risk<=1: raise HTTPException(400,"minimum_anomaly_risk must be in [0,1]")
    c=conn()
    with c.cursor() as cur:
        cur.execute("INSERT INTO regression_tests(machine_id,description,timestamp_range,minimum_anomaly_risk) VALUES(%s,%s,tstzrange(%s,%s,'[]'),%s) RETURNING id",(machine(body.machine_id),body.description,body.start,body.end,body.minimum_anomaly_risk)); rid=cur.fetchone()[0]
    c.commit(); c.close(); return {"id":rid}

@app.post("/api/backfill")
def run_backfill(body:BackfillBody,admin:User=Depends(require_admin)):
    c=conn()
    try: return backfill.run(c,body.start,body.end,body.mode,machine(body.machine_id) if body.machine_id else None)
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()

@app.get("/api/retrain/status")
def retrain_status(user:User=Depends(current_user)):
    c=conn(); due,status=retrain_service.should_retrain(c); c.close(); return {**status,"due":due}
@app.post("/api/retrain/trigger")
def retrain_now(admin:User=Depends(require_admin)):
    c=conn()
    try: return retrain_service.run_shadow_retrain(c,force=True)
    finally: c.close()

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

@app.get("/api/models/{version_id}/calibrations")
def list_calibrations(version_id:str,machine_id:str|None=None,user:User=Depends(current_user)):
    selected_machine=machine(machine_id)
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT id,version_id,machine_id,source_rows,source_description,created_at,created_by FROM model_calibrations WHERE version_id=%s AND machine_id=%s ORDER BY created_at DESC",(version_id,selected_machine))
        rows=cur.fetchall()
        cur.execute("SELECT calibration_id AS active_calibration_id FROM machine_model_calibrations WHERE version_id=%s AND machine_id=%s",(version_id,selected_machine))
        row=cur.fetchone()
        active_calibration_id=row["active_calibration_id"] if row else None
    c.close(); return {"items":rows,"active_calibration_id":active_calibration_id}
@app.post("/api/models/{version_id}/recalibrate")
def recalibrate_model(version_id:str,body:RecalibrateBody,admin:User=Depends(require_admin)):
    if body.hours<=0: raise HTTPException(400,"hours must be > 0")
    if body.min_rows<1: raise HTTPException(400,"min_rows must be >= 1")
    if body.source not in recalibrate_service.SOURCES: raise HTTPException(400,f"source must be one of {recalibrate_service.SOURCES}")
    c=conn()
    try: return recalibrate_service.run(c,version_id,body.hours,body.min_rows,admin.username,source=body.source,machine_id=machine(body.machine_id))
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()
@app.post("/api/models/{version_id}/calibration/activate")
def activate_calibration(version_id:str,body:CalibrationActivateBody,machine_id:str|None=None,admin:User=Depends(require_admin)):
    selected_machine=machine(machine_id)
    c=conn()
    try: model_registry.set_calibration(c,version_id,body.calibration_id,selected_machine)
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()
    return {"version_id":version_id,"machine_id":selected_machine,"active_calibration_id":body.calibration_id}
@app.delete("/api/models/{version_id}/calibration/{calibration_id}")
def delete_calibration(version_id:str,calibration_id:int,machine_id:str|None=None,admin:User=Depends(require_admin)):
    selected_machine=machine(machine_id)
    c=conn()
    try: model_registry.delete_calibration(c,version_id,calibration_id,selected_machine)
    except ValueError as e: raise HTTPException(400,str(e))
    finally: c.close()
    return {"deleted":calibration_id}

@app.get("/api/history")
def history(limit:int=50,offset:int=0,level:str|None=None,machine_id:str|None=None,user:User=Depends(current_user)):
    limit=max(1,min(limit,500)); offset=max(0,offset)
    selected_machine=machine(machine_id); conditions=["p.machine_id=%s"]; args:list=[selected_machine]
    if level:
        if level not in ("OK","WARN","CRITICAL"): raise HTTPException(400,"level must be OK, WARN, or CRITICAL")
        conditions.append("p.maintenance_level=%s"); args.append(level)
    where="WHERE "+" AND ".join(conditions)
    # near-miss eligibility window, same "row count" convention /api/near-miss
    # uses for its regr_slope() window (see that endpoint for why hours*60).
    nm_window=max(2,6*60)
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM spindle_predictions p {where}",args); total=cur.fetchone()["total"]
        cur.execute(f"""
          WITH nm AS (
            SELECT id, maintenance_level,
                   regr_slope(anomaly_score, extract(epoch from tick_timestamp))
                     OVER (ORDER BY tick_timestamp ROWS BETWEEN %s PRECEDING AND CURRENT ROW) AS slope
            FROM spindle_predictions
            WHERE machine_id=%s
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
          ORDER BY p.tick_timestamp DESC LIMIT %s OFFSET %s""",[nm_window,selected_machine]+args+[limit,offset]); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.get("/api/near-miss")
def near_miss(hours:int=6,limit:int=50,offset:int=0,status:str="pending",machine_id:str|None=None,user:User=Depends(current_user)):
    if status not in ("pending","acknowledged","flagged"): raise HTTPException(400,"status must be pending, acknowledged, or flagged")
    limit=max(1,min(limit,500)); offset=max(0,offset); window=max(2,hours*60)
    selected_machine=machine(machine_id)
    base_cte="""WITH x AS (
          SELECT id, machine_id, tick_timestamp, model_version, anomaly_score, health_state, maintenance_level, raw_reading,
                 regr_slope(anomaly_score, extract(epoch from tick_timestamp)) OVER (ORDER BY tick_timestamp ROWS BETWEEN %s PRECEDING AND CURRENT ROW) slope
          FROM spindle_predictions WHERE machine_id=%s)
          SELECT x.*, COALESCE(nmr.status,'pending') AS review_status, nmr.reviewed_by, nmr.reviewed_at
          FROM x LEFT JOIN near_miss_reviews nmr ON nmr.prediction_id=x.id
          WHERE x.maintenance_level='OK' AND x.slope IS NOT NULL AND x.slope < 0
            AND COALESCE(nmr.status,'pending')=%s"""
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM ({base_cte}) t",(window,selected_machine,status)); total=cur.fetchone()["total"]
        cur.execute(base_cte+" ORDER BY x.health_state ASC LIMIT %s OFFSET %s",(window,selected_machine,status,limit,offset)); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.post("/api/near-miss/{prediction_id}/review")
def review_near_miss(prediction_id:int,body:ReviewBody,user:User=Depends(current_user)):
    if body.decision not in ("acknowledged","flagged"): raise HTTPException(400,"decision must be acknowledged or flagged")
    c=conn()
    with c.cursor() as cur:
        cur.execute("SELECT id, machine_id, tick_timestamp, anomaly_score FROM spindle_predictions WHERE id=%s",(prediction_id,)); row=cur.fetchone()
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
        # much risk in this time window, or promotion is blocked.
        regression_test_id=None
        if body.decision=="flagged":
            _, machine_id, tick_ts, anomaly_score=row
            hours=float(runtime_config.get("NEAR_MISS_REGRESSION_WINDOW_HOURS",1))
            min_risk=max(0.05,min(0.95,float(anomaly_score) if anomaly_score is not None else 0.5))
            cur.execute("""INSERT INTO regression_tests(machine_id,description,timestamp_range,minimum_anomaly_risk)
                           VALUES(%s,%s,tstzrange(%s-(%s||' hours')::interval,%s+(%s||' hours')::interval,'[]'),%s)
                           RETURNING id""",
                        (machine_id,f"Near-miss flagged by {user.username} on prediction #{prediction_id}",tick_ts,hours,tick_ts,hours,min_risk))
            regression_test_id=cur.fetchone()[0]
    c.commit(); c.close(); return {"prediction_id":prediction_id,"status":body.decision,"regression_test_id":regression_test_id}

@app.websocket("/ws/live")
async def live(ws:WebSocket):
    selected_machine=machine(ws.query_params.get("machine_id"))
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
