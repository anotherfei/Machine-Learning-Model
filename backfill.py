"""Historical re-prediction using the exact production SpindleMonitor implementation."""
from __future__ import annotations
import json
import pandas as pd
import psycopg2.extras
import psycopg2.sql as sql
import artifact_utils, config, db, model_registry
from isolation_forest import AnomalyScorer
from predict_realtime import SpindleMonitor


def run(conn, start, end, mode="repredict", machine_id=None):
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    if end <= start: raise ValueError("end must be after start")
    if mode=="skip": return {"mode":"skip","processed":0}
    if mode!="repredict": raise ValueError("mode must be repredict or skip")
    warmup=start-pd.Timedelta(minutes=max(config.TREND_LOOKBACK_MINUTES, config.WINDOW_SIZE+config.KALMAN_INIT_SAMPLES))
    table=db.get_table_name(); c=db.get_db_columns(); cols=[c["timestamp"]]+([c["machine_id"]] if c["machine_id"] else [])+c["sensor_cols"]
    where=sql.SQL("{ts} >= %s AND {ts} <= %s").format(ts=sql.Identifier(c["timestamp"])); args=[warmup.to_pydatetime(),end.to_pydatetime()]
    if machine_id and c["machine_id"]:
        where += sql.SQL(" AND {machine} = %s").format(machine=sql.Identifier(c["machine_id"])); args.append(machine_id)
    elif machine_id and machine_id != db.DEFAULT_MACHINE_ID:
        return {"mode":"repredict","processed":0,"machine_id":machine_id,"start":start.isoformat(),"end":end.isoformat()}
    order=sql.SQL("{ts}, {machine}").format(ts=sql.Identifier(c["timestamp"]),machine=sql.Identifier(c["machine_id"])) if c["machine_id"] else sql.Identifier(c["timestamp"])
    q=sql.SQL("SELECT {cols} FROM {table} WHERE {where} ORDER BY {order}").format(
        cols=sql.SQL(',').join(sql.Identifier(x) for x in cols), table=sql.Identifier(table), where=where, order=order)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(q,args); rows=cur.fetchall()
    base_scorer,feature_cols,metadata=artifact_utils.load_artifacts(use_local_calibration=False)
    version_id=metadata.get("version_id",metadata.get("trained_at","unversioned"))
    monitors={}; written=0; machine_ids=set()
    for row in rows:
        row_machine_id=db.row_machine_id(row,c); machine_ids.add(row_machine_id)
        monitor=monitors.get(row_machine_id)
        if monitor is None:
            calibration=model_registry.machine_calibration(conn,version_id,row_machine_id)
            machine_scorer=AnomalyScorer.from_calibration(base_scorer.model,calibration) if calibration else base_scorer
            monitor=SpindleMonitor(machine_scorer,feature_cols,metadata); monitors[row_machine_id]=monitor
        tick=pd.Timestamp(row[c["timestamp"]]); reading={col:float(row[c["by_config_name"][col]]) for col in config.RAW_SENSOR_COLS}
        result=monitor.update(reading)
        if result is None or tick < start: continue
        rec=result["maintenance"]
        raw={k:result[k] for k in config.RAW_SENSOR_COLS}
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO spindle_predictions
              (machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,trend_slope_per_day,remaining_days,failure_probability,
               maintenance_level,maintenance_reason,maintenance_trigger,top_contributors,is_backfill)
              VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,TRUE)
              ON CONFLICT(machine_id,tick_timestamp,model_version) DO UPDATE SET
                raw_reading=EXCLUDED.raw_reading, anomaly_score=EXCLUDED.anomaly_score, health_raw=EXCLUDED.health_raw,
                health_state=EXCLUDED.health_state, trend_slope_per_day=EXCLUDED.trend_slope_per_day, remaining_days=EXCLUDED.remaining_days,
                failure_probability=EXCLUDED.failure_probability, maintenance_level=EXCLUDED.maintenance_level,
                maintenance_reason=EXCLUDED.maintenance_reason, maintenance_trigger=EXCLUDED.maintenance_trigger,
                top_contributors=EXCLUDED.top_contributors, is_backfill=TRUE""",
              (row_machine_id,tick,result["model_version"],json.dumps(raw),result["anomaly_score"],result["health_raw"],result["health_state"],result["trend_slope_per_day"],result["remaining_days"],json.dumps(result["failure_probability"]),rec["level"],rec["reason"],rec.get("trigger","none"),json.dumps(result.get("top_contributors"))))
        written+=1
    conn.commit(); return {"mode":"repredict","processed":written,"machine_ids":sorted(machine_ids),"start":start.isoformat(),"end":end.isoformat()}
