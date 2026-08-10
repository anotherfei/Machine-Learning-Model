"""Long-running production worker: polling, prediction persistence, alerts, hot reload."""
from __future__ import annotations
import json
import select
import sys
import time
import psycopg2.extras

import config
import db
import db_schema
import model_registry
import predict_realtime
import runtime_config

POLL_SECONDS = 60


def _listen_conn():
    c = db.get_connection(); c.set_isolation_level(0)
    with c.cursor() as cur: cur.execute("LISTEN model_changed; LISTEN config_changed; LISTEN env_changed;")
    return c


def _persist(conn, result, tick_ts):
    raw = {k: result[k] for k in config.RAW_SENSOR_COLS}
    rec = result["maintenance"]
    with conn.cursor() as cur:
        cur.execute("""INSERT INTO spindle_predictions
            (tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,trend_slope_per_day,remaining_days,
             failure_probability,maintenance_level,maintenance_reason,maintenance_trigger,top_contributors)
             VALUES(%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb)""",
            (tick_ts,result["model_version"],json.dumps(raw),result["anomaly_score"],result["health_raw"],result["health_state"],
             result["trend_slope_per_day"],result["remaining_days"],json.dumps(result["failure_probability"]),rec["level"],rec["reason"],rec.get("trigger","none"),json.dumps(result.get("top_contributors"))))
        if rec["level"] in ("WARN","CRITICAL"):
            cur.execute("""INSERT INTO alerts(tick_timestamp,model_version,trigger,level,health_state,anomaly_score,raw_reading,feature_vector)
                           VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)""",
                        (tick_ts,result["model_version"],rec.get("trigger","none"),rec["level"],result["health_state"],result["anomaly_score"],json.dumps(raw),json.dumps(result.get("feature_vector"))))
        payload=json.dumps({"timestamp": str(tick_ts), "model_version": result["model_version"], "health_state": result["health_state"], "anomaly_score": result["anomaly_score"], "maintenance": rec, **raw})
        cur.execute("NOTIFY prediction_tick, %s", (payload,))
    conn.commit()


def main():
    conn=db.get_connection(); db_schema.migrate(conn)
    listener=_listen_conn()
    monitor=predict_realtime.SpindleMonitor()
    table=db.get_table_name(); cols=db.get_db_columns(); last_seen=None
    print("Spindle Condition Monitoring worker started")
    while True:
        # Notifications are applied only at a tick boundary.
        if select.select([listener],[],[],0)[0]:
            listener.poll()
            while listener.notifies:
                note=listener.notifies.pop(0)
                if note.channel=="config_changed": runtime_config.load_from_db(conn)
                elif note.channel=="model_changed":
                    sc, fc, md=predict_realtime.reload_model(); monitor.scorer=sc; monitor.feature_cols=fc; monitor.metadata=md
                elif note.channel=="env_changed":
                    print("Environment changed; graceful worker restart requested")
                    return 75
        rows=db.fetch_new_rows(conn,table,since=last_seen)
        for row in rows:
            reading={col:float(row[cols["by_config_name"][col]]) for col in config.RAW_SENSOR_COLS}
            tick_ts=row[cols["timestamp"]]; result=monitor.update(reading); last_seen=tick_ts
            if result is not None: _persist(conn,result,tick_ts)
        time.sleep(POLL_SECONDS)

if __name__=="__main__": sys.exit(main())
