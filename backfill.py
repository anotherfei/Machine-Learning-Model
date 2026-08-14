"""Historical re-prediction using the exact production SpindleMonitor implementation."""
from __future__ import annotations
import json
import pandas as pd
import psycopg2.extras
import psycopg2.sql as sql
import artifact_utils, config, db, machine_normalization, model_registry, operating_state, runtime_config
from isolation_forest import AnomalyScorer
from predict_realtime import SpindleMonitor


def run(conn, start, end, mode="repredict", machine_id=None):
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    if end <= start: raise ValueError("end must be after start")
    if mode=="skip":
        return {
            "mode":"skip","processed":0,"skipped_reviewed":0,"machine_ids":[],
            "start":start.isoformat(),"end":end.isoformat(),
        }
    if mode!="repredict": raise ValueError("mode must be repredict or skip")
    warmup=start-pd.Timedelta(minutes=max(
        int(runtime_config.get("TREND_LOOKBACK_MINUTES", config.TREND_LOOKBACK_MINUTES)),
        config.WINDOW_SIZE + int(runtime_config.get("KALMAN_INIT_SAMPLES", config.KALMAN_INIT_SAMPLES)),
    ))
    table=db.get_table_name(); c=db.get_db_columns(); cols=[c["timestamp"]]+([c["machine_id"]] if c["machine_id"] else [])+c["sensor_cols"]
    where=sql.SQL("{ts} >= %s AND {ts} <= %s").format(ts=sql.Identifier(c["timestamp"])); args=[warmup.to_pydatetime(),end.to_pydatetime()]
    if machine_id and c["machine_id"]:
        where += sql.SQL(" AND {machine} = %s").format(
            machine=sql.Identifier(c["machine_id"])
        ); args.append(machine_id)
    elif machine_id and machine_id != db.DEFAULT_MACHINE_ID:
        return {
            "mode":"repredict","processed":0,"skipped_reviewed":0,
            "machine_ids":[],"start":start.isoformat(),"end":end.isoformat(),
        }
    order=sql.SQL("{ts}, {machine}").format(ts=sql.Identifier(c["timestamp"]),machine=sql.Identifier(c["machine_id"])) if c["machine_id"] else sql.Identifier(c["timestamp"])
    q=sql.SQL("SELECT {cols} FROM {table} WHERE {where} ORDER BY {order}").format(
        cols=sql.SQL(',').join(sql.Identifier(x) for x in cols), table=sql.Identifier(table), where=where, order=order)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(q,args); rows=cur.fetchall()
    base_scorer,feature_cols,metadata=artifact_utils.load_artifacts()
    normalizers=artifact_utils.load_machine_feature_normalizers()
    version_id=metadata.get("version_id",metadata.get("trained_at","unversioned"))
    monitors={}; detectors={}; written=0; machine_ids=set(); skipped_reviewed=0
    # Learn thresholds from the whole fetched warm-up/range first. Each
    # detector remains machine-local and refuses to classify STOPPED if the
    # history does not contain two clearly distinct regimes.
    for row in rows:
        row_machine_id=db.row_machine_id(row,c)
        detector=detectors.setdefault(row_machine_id,operating_state.OperatingStateDetector())
        detector.observe_history(db.canonical_sensor_reading(row,c))
    for detector in detectors.values():
        detector.fit_history()
    for row in rows:
        row_machine_id=db.row_machine_id(row,c); machine_ids.add(row_machine_id)
        reading=db.canonical_sensor_reading(row,c)
        detector=detectors[row_machine_id]
        state=detector.update(reading)
        if state.state == "STOPPED":
            if state.changed: monitors.pop(row_machine_id,None)
            continue
        if state.state == "SENSOR_FAULT":
            continue
        if state.low_motion:
            continue
        monitor=monitors.get(row_machine_id)
        if monitor is None:
            calibration=model_registry.machine_calibration(conn,version_id,row_machine_id)
            if calibration is None:
                raise ValueError(
                    f"No automatic condition anchor exists for {row_machine_id!r}. "
                    "Retrain with this machine included before backfilling it."
                )
            normalizer=machine_normalization.for_machine(normalizers,row_machine_id)
            machine_scorer=AnomalyScorer.from_calibration(base_scorer.model,calibration)
            monitor=SpindleMonitor(
                machine_scorer,feature_cols,metadata,
                feature_normalizer_override=normalizer,machine_id=row_machine_id,
            ); monitors[row_machine_id]=monitor
        tick=pd.Timestamp(row[c["timestamp"]]); numeric_reading={col:float(reading[col]) for col in config.RAW_SENSOR_COLS}
        result=monitor.update(numeric_reading,timestamp=tick)
        if result is None or state.state == "STARTING" or tick < start: continue
        rec=result["maintenance"]
        raw={k:result[k] for k in config.RAW_SENSOR_COLS}
        with conn.cursor() as cur:
            # A reviewed alert/near-miss is a human decision about the exact
            # prediction evidence that existed at review time. Do not silently
            # rewrite that evidence during a later historical replay.
            cur.execute(
                """SELECT status FROM alerts
                   WHERE machine_id=%s AND tick_timestamp=%s AND model_version=%s
                   FOR UPDATE""",
                (row_machine_id,tick,result["model_version"]),
            )
            alert_statuses=[item[0] for item in cur.fetchall()]
            cur.execute(
                """SELECT nmr.id FROM near_miss_reviews nmr
                   JOIN spindle_predictions p ON p.id=nmr.prediction_id
                   WHERE p.machine_id=%s AND p.tick_timestamp=%s AND p.model_version=%s
                   FOR UPDATE OF nmr""",
                (row_machine_id,tick,result["model_version"]),
            )
            reviewed_near_miss=cur.fetchone() is not None
            if any(status != "pending" for status in alert_statuses) or reviewed_near_miss:
                skipped_reviewed+=1
                continue
            cur.execute("""INSERT INTO spindle_predictions
              (machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,trend_slope_per_day,remaining_days,failure_probability,
               maintenance_level,maintenance_reason,maintenance_trigger,top_contributors,is_backfill)
              VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,TRUE)
              ON CONFLICT(machine_id,tick_timestamp,model_version) DO UPDATE SET
                raw_reading=EXCLUDED.raw_reading, anomaly_score=EXCLUDED.anomaly_score, health_raw=EXCLUDED.health_raw,
                health_state=EXCLUDED.health_state, trend_slope_per_day=EXCLUDED.trend_slope_per_day, remaining_days=EXCLUDED.remaining_days,
                failure_probability=EXCLUDED.failure_probability, maintenance_level=EXCLUDED.maintenance_level,
                maintenance_reason=EXCLUDED.maintenance_reason, maintenance_trigger=EXCLUDED.maintenance_trigger,
                top_contributors=EXCLUDED.top_contributors, is_backfill=TRUE
              RETURNING id""",
              (row_machine_id,tick,result["model_version"],json.dumps(raw),result["anomaly_score"],result["health_raw"],result["health_state"],result["trend_slope_per_day"],result["remaining_days"],json.dumps(result["failure_probability"]),rec["level"],rec["reason"],rec.get("trigger","none"),json.dumps(result.get("top_contributors"))))
            cur.fetchone()  # consume RETURNING id; row identity is preserved by the upsert
            if rec["level"] in ("WARN","CRITICAL"):
                cur.execute(
                    """UPDATE alerts SET trigger=%s,level=%s,health_state=%s,anomaly_score=%s,
                              raw_reading=%s::jsonb,feature_vector=%s::jsonb
                       WHERE machine_id=%s AND tick_timestamp=%s AND model_version=%s
                         AND status='pending'""",
                    (rec.get("trigger","none"),rec["level"],result["health_state"],result["anomaly_score"],
                     json.dumps(raw),json.dumps(result.get("feature_vector")),row_machine_id,tick,
                     result["model_version"]),
                )
                if cur.rowcount == 0:
                    cur.execute(
                        """INSERT INTO alerts
                           (machine_id,tick_timestamp,model_version,trigger,level,health_state,
                            anomaly_score,raw_reading,feature_vector)
                           VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)""",
                        (row_machine_id,tick,result["model_version"],rec.get("trigger","none"),
                         rec["level"],result["health_state"],result["anomaly_score"],
                         json.dumps(raw),json.dumps(result.get("feature_vector"))),
                    )
            else:
                # Only pending machine-generated alerts are replaceable.
                # Reviewed rows were excluded above and reference candidates
                # cascade only from reviewed confirmed-normal alerts.
                cur.execute(
                    """DELETE FROM alerts WHERE machine_id=%s AND tick_timestamp=%s
                         AND model_version=%s AND status='pending'""",
                    (row_machine_id,tick,result["model_version"]),
                )
        written+=1
    conn.commit(); return {
        "mode":"repredict","processed":written,"skipped_reviewed":skipped_reviewed,
        "machine_ids":sorted(machine_ids),"start":start.isoformat(),"end":end.isoformat(),
    }
