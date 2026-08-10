"""Historical re-prediction using the exact production SpindleMonitor implementation."""
from __future__ import annotations
import json
import pandas as pd
import psycopg2.extras
import psycopg2.sql as sql
import config, db
from predict_realtime import SpindleMonitor


def run(conn, start, end, mode="repredict"):
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    if end <= start: raise ValueError("end must be after start")
    if mode=="skip": return {"mode":"skip","processed":0}
    if mode!="repredict": raise ValueError("mode must be repredict or skip")
    warmup=start-pd.Timedelta(minutes=max(config.TREND_LOOKBACK_MINUTES, config.WINDOW_SIZE+config.KALMAN_INIT_SAMPLES))
    table=db.get_table_name(); c=db.get_db_columns(); cols=[c["timestamp"]]+c["sensor_cols"]
    q=sql.SQL("SELECT {cols} FROM {table} WHERE {ts} >= %s AND {ts} <= %s ORDER BY {ts}").format(
        cols=sql.SQL(',').join(sql.Identifier(x) for x in cols), table=sql.Identifier(table), ts=sql.Identifier(c["timestamp"]))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(q,(warmup.to_pydatetime(),end.to_pydatetime())); rows=cur.fetchall()
    monitor=SpindleMonitor(); written=0
    for row in rows:
        tick=pd.Timestamp(row[c["timestamp"]]); reading={col:float(row[c["by_config_name"][col]]) for col in config.RAW_SENSOR_COLS}
        result=monitor.update(reading)
        if result is None or tick < start: continue
        rec=result["maintenance"]
        raw={k:result[k] for k in config.RAW_SENSOR_COLS}
        with conn.cursor() as cur:
            cur.execute("""INSERT INTO spindle_predictions
              (tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,trend_slope_per_day,remaining_days,failure_probability,
               maintenance_level,maintenance_reason,maintenance_trigger,top_contributors,is_backfill)
              VALUES(%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,TRUE)
              ON CONFLICT(tick_timestamp,model_version) DO UPDATE SET
                raw_reading=EXCLUDED.raw_reading, anomaly_score=EXCLUDED.anomaly_score, health_raw=EXCLUDED.health_raw,
                health_state=EXCLUDED.health_state, trend_slope_per_day=EXCLUDED.trend_slope_per_day, remaining_days=EXCLUDED.remaining_days,
                failure_probability=EXCLUDED.failure_probability, maintenance_level=EXCLUDED.maintenance_level,
                maintenance_reason=EXCLUDED.maintenance_reason, maintenance_trigger=EXCLUDED.maintenance_trigger,
                top_contributors=EXCLUDED.top_contributors, is_backfill=TRUE""",
              (tick,result["model_version"],json.dumps(raw),result["anomaly_score"],result["health_raw"],result["health_state"],result["trend_slope_per_day"],result["remaining_days"],json.dumps(result["failure_probability"]),rec["level"],rec["reason"],rec.get("trigger","none"),json.dumps(result.get("top_contributors"))))
        written+=1
    conn.commit(); return {"mode":"repredict","processed":written,"start":start.isoformat(),"end":end.isoformat()}
