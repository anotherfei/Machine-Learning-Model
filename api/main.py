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

import db
import db_schema
import env_manager
import model_registry
import retrain_service
import runtime_config
import backfill
from api.auth import COOKIE, User, current_user, hash_password, issue_cookie, require_admin, verify_password

app=FastAPI(title="Spindle Condition Monitoring API", version="1.0")
app.add_middleware(CORSMiddleware, allow_origins=[os.getenv("FRONTEND_ORIGIN","http://localhost:5173")], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
_login_attempts=defaultdict(deque)
scheduler=BackgroundScheduler(daemon=True)


def conn(): return db.get_connection()


def _bootstrap():
    c=conn(); db_schema.migrate(c)
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
            import artifact_utils
            ts=artifact_utils.load_reference_timestamps(); signature=model_registry.reference_signature(ts if ts is not None else [])
            cur.execute("INSERT INTO model_versions(version_id,artifact_path,reference_signature,status,promoted_at,promoted_by) VALUES(%s,%s,%s,'active',now(),'bootstrap')",(vid,path,signature))
    c.commit(); c.close()


def _scheduled_retrain():
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
    minimum_anomaly_risk:float=0.6
class BackfillBody(BaseModel): start:str; end:str; mode:str="repredict"

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

@app.get("/api/alerts")
def alerts(status:str="pending",trigger:str|None=None,limit:int=50,offset:int=0,user:User=Depends(current_user)):
    limit=max(1,min(limit,500)); offset=max(0,offset); c=conn()
    where=["status=%s"]; args=[status]
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
        cur.execute("SELECT tick_timestamp FROM alerts WHERE id=%s",(alert_id,)); row=cur.fetchone()
        if not row: c.close(); raise HTTPException(404,"Alert not found")
        ts=row[0]
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""SELECT tick_timestamp,health_state,anomaly_score,maintenance_level FROM spindle_predictions
                       WHERE tick_timestamp BETWEEN %s-(%s||' hours')::interval AND %s+(%s||' hours')::interval
                       ORDER BY tick_timestamp""",(ts,hours,ts,hours)); rows=cur.fetchall()
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
        cur.execute("INSERT INTO regression_tests(description,timestamp_range,minimum_anomaly_risk) VALUES(%s,tstzrange(%s,%s,'[]'),%s) RETURNING id",(body.description,body.start,body.end,body.minimum_anomaly_risk)); rid=cur.fetchone()[0]
    c.commit(); c.close(); return {"id":rid}

@app.post("/api/backfill")
def run_backfill(body:BackfillBody,admin:User=Depends(require_admin)):
    c=conn()
    try: return backfill.run(c,body.start,body.end,body.mode)
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

@app.get("/api/history")
def history(limit:int=50,offset:int=0,level:str|None=None,user:User=Depends(current_user)):
    limit=max(1,min(limit,500)); offset=max(0,offset)
    where=""; args:list=[]
    if level:
        if level not in ("OK","WARN","CRITICAL"): raise HTTPException(400,"level must be OK, WARN, or CRITICAL")
        where="WHERE p.maintenance_level=%s"; args=[level]
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM spindle_predictions p {where}",args); total=cur.fetchone()["total"]
        cur.execute(f"""
          SELECT p.*,
                 a.status AS alert_status, a.level AS alert_level, a.trigger AS alert_trigger,
                 a.reviewed_by AS alert_reviewed_by, a.reviewed_at AS alert_reviewed_at,
                 nmr.status AS near_miss_status, nmr.reviewed_by AS near_miss_reviewed_by, nmr.reviewed_at AS near_miss_reviewed_at
          FROM spindle_predictions p
          LEFT JOIN LATERAL (
            SELECT status, level, trigger, reviewed_by, reviewed_at FROM alerts a
            WHERE a.tick_timestamp=p.tick_timestamp AND a.model_version=p.model_version
            ORDER BY a.id DESC LIMIT 1
          ) a ON true
          LEFT JOIN near_miss_reviews nmr ON nmr.prediction_id=p.id
          {where}
          ORDER BY p.tick_timestamp DESC LIMIT %s OFFSET %s""",args+[limit,offset]); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.get("/api/near-miss")
def near_miss(hours:int=6,limit:int=50,offset:int=0,status:str="pending",user:User=Depends(current_user)):
    if status not in ("pending","acknowledged","flagged"): raise HTTPException(400,"status must be pending, acknowledged, or flagged")
    limit=max(1,min(limit,500)); offset=max(0,offset); window=max(2,hours*60)
    base_cte="""WITH x AS (
          SELECT id, tick_timestamp, model_version, anomaly_score, health_state, maintenance_level, raw_reading,
                 regr_slope(anomaly_score, extract(epoch from tick_timestamp)) OVER (ORDER BY tick_timestamp ROWS BETWEEN %s PRECEDING AND CURRENT ROW) slope
          FROM spindle_predictions)
          SELECT x.*, COALESCE(nmr.status,'pending') AS review_status, nmr.reviewed_by, nmr.reviewed_at
          FROM x LEFT JOIN near_miss_reviews nmr ON nmr.prediction_id=x.id
          WHERE x.maintenance_level='OK' AND x.slope IS NOT NULL AND x.slope < 0
            AND COALESCE(nmr.status,'pending')=%s"""
    c=conn()
    with c.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(f"SELECT count(*) AS total FROM ({base_cte}) t",(window,status)); total=cur.fetchone()["total"]
        cur.execute(base_cte+" ORDER BY x.health_state ASC LIMIT %s OFFSET %s",(window,status,limit,offset)); rows=cur.fetchall()
    c.close(); return {"items":rows,"total":total,"limit":limit,"offset":offset}

@app.post("/api/near-miss/{prediction_id}/review")
def review_near_miss(prediction_id:int,body:ReviewBody,user:User=Depends(current_user)):
    if body.decision not in ("acknowledged","flagged"): raise HTTPException(400,"decision must be acknowledged or flagged")
    c=conn()
    with c.cursor() as cur:
        cur.execute("SELECT id, tick_timestamp, anomaly_score FROM spindle_predictions WHERE id=%s",(prediction_id,)); row=cur.fetchone()
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
            _, tick_ts, anomaly_score=row
            hours=float(runtime_config.get("NEAR_MISS_REGRESSION_WINDOW_HOURS",1))
            min_risk=max(0.05,min(0.95,float(anomaly_score) if anomaly_score is not None else 0.5))
            cur.execute("""INSERT INTO regression_tests(description,timestamp_range,minimum_anomaly_risk)
                           VALUES(%s,tstzrange(%s-(%s||' hours')::interval,%s+(%s||' hours')::interval,'[]'),%s)
                           RETURNING id""",
                        (f"Near-miss flagged by {user.username} on prediction #{prediction_id}",tick_ts,hours,tick_ts,hours,min_risk))
            regression_test_id=cur.fetchone()[0]
    c.commit(); c.close(); return {"prediction_id":prediction_id,"status":body.decision,"regression_test_id":regression_test_id}

@app.websocket("/ws/live")
async def live(ws:WebSocket):
    await ws.accept(); c=conn(); c.set_isolation_level(0)
    with c.cursor() as cur: cur.execute("LISTEN prediction_tick")
    try:
        while True:
            await asyncio.sleep(.5)
            if select.select([c],[],[],0)[0]:
                c.poll()
                while c.notifies:
                    note=c.notifies.pop(0); await ws.send_text(note.payload)
    except WebSocketDisconnect: pass
    finally: c.close()
