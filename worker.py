"""Long-running production worker: polling, prediction persistence, alerts, hot reload."""
from __future__ import annotations
import json
import math
import select
import sys
import time
import psycopg2.extras

import config
import db
import db_schema
import model_registry
import machine_normalization
import operating_state
import predict_realtime
import runtime_config
from isolation_forest import AnomalyScorer

MONITOR_RESET_KEYS = {
    "HEALTH_SENSITIVITY_STD",
    "KALMAN_INIT_SAMPLES",
    "TREND_LOOKBACK_MINUTES",
}


def _json_number(value):
    try:
        number=float(value)
        return number if math.isfinite(number) else None
    except (TypeError,ValueError):
        return None


def _new_monitor(conn, machine_id):
    if predict_realtime.scorer is None:
        predict_realtime.reload_model()
    base=predict_realtime.scorer; feature_cols=predict_realtime.feature_cols; metadata=predict_realtime.metadata or {}
    version_id=metadata.get("version_id",metadata.get("trained_at","unversioned"))
    calibration=model_registry.machine_calibration(conn,version_id,machine_id)
    if calibration is None:
        raise ValueError(
            f"No automatic condition anchor exists for commissioned machine {machine_id!r}. "
            "Retrain the shared model with this machine included."
        )
    normalizer=machine_normalization.for_machine(
        predict_realtime.machine_feature_normalizers,machine_id
    )
    machine_scorer=AnomalyScorer.from_calibration(base.model,calibration)
    return predict_realtime.SpindleMonitor(
        machine_scorer,feature_cols,metadata,
        feature_normalizer_override=normalizer,machine_id=machine_id,
    )


def _listen_conn():
    c = db.get_connection(); c.set_isolation_level(0)
    with c.cursor() as cur: cur.execute("LISTEN model_changed; LISTEN config_changed; LISTEN env_changed;")
    return c


def _persist_tick(conn, machine_id, reading, state, result, tick_ts):
    raw = {
        key:_json_number(value)
        for key, value in reading.items()
    }
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO machine_runtime_state
               (machine_id,operating_state,reason,confidence,activity_score,stop_threshold,run_threshold,
                tick_timestamp,state_changed_at)
               VALUES(%s,%s,%s,%s,%s,%s,%s,%s,%s)
               ON CONFLICT(machine_id) DO UPDATE SET
                 operating_state=EXCLUDED.operating_state,
                 reason=EXCLUDED.reason,
                 confidence=EXCLUDED.confidence,
                 activity_score=EXCLUDED.activity_score,
                 stop_threshold=EXCLUDED.stop_threshold,
                 run_threshold=EXCLUDED.run_threshold,
                 tick_timestamp=EXCLUDED.tick_timestamp,
                 state_changed_at=CASE
                   WHEN machine_runtime_state.operating_state IS DISTINCT FROM EXCLUDED.operating_state
                   THEN EXCLUDED.tick_timestamp ELSE machine_runtime_state.state_changed_at END,
                 updated_at=now()""",
            (
                machine_id,state["state"],state["reason"],state["confidence"],state["activity_score"],
                state["stop_threshold"],state["run_threshold"],tick_ts,tick_ts,
            ),
        )

        inserted = False
        if result is not None:
            rec = result["maintenance"]
            cur.execute("""INSERT INTO spindle_predictions
                (machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,trend_slope_per_day,remaining_days,
                 failure_probability,maintenance_level,maintenance_reason,maintenance_trigger,top_contributors)
                 VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb)
                 ON CONFLICT(machine_id,tick_timestamp,model_version) DO NOTHING RETURNING id""",
                (machine_id,tick_ts,result["model_version"],json.dumps(raw),result["anomaly_score"],result["health_raw"],result["health_state"],
                 result["trend_slope_per_day"],result["remaining_days"],json.dumps(result["failure_probability"]),rec["level"],rec["reason"],rec.get("trigger","none"),json.dumps(result.get("top_contributors"))))
            inserted = cur.fetchone() is not None
            if inserted and rec["level"] in ("WARN","CRITICAL"):
                cur.execute("""INSERT INTO alerts(machine_id,tick_timestamp,model_version,trigger,level,health_state,anomaly_score,raw_reading,feature_vector)
                               VALUES(%s,%s,%s,%s,%s,%s,%s,%s::jsonb,%s::jsonb)""",
                            (machine_id,tick_ts,result["model_version"],rec.get("trigger","none"),rec["level"],result["health_state"],result["anomaly_score"],json.dumps(raw),json.dumps(result.get("feature_vector"))))

        payload={
            "machine_id":machine_id,
            "timestamp":str(tick_ts),
            "prediction_timestamp":str(tick_ts) if result is not None else None,
            "prediction_lag_seconds":0.0 if result is not None else None,
            "operating_state":state["state"],
            "operating_state_reason":state["reason"],
            "operating_state_confidence":state["confidence"],
            "operating_state_changed":state["changed"],
            "operating_state_activity":state["activity_score"],
            "operating_state_stop_threshold":state["stop_threshold"],
            "operating_state_run_threshold":state["run_threshold"],
            "prediction_available":result is not None,
            **raw,
        }
        if result is None and state["state"] in ("RUNNING","UNKNOWN"):
            payload["prediction_wait_reason"]=(
                "The inference pipeline is warming up; condition and maintenance values "
                "will appear after enough valid running readings are available."
            )
        if result is not None:
            payload.update({
                "model_version":result["model_version"],
                "health_state":result["health_state"],
                "anomaly_score":result["anomaly_score"],
                "maintenance":result["maintenance"],
            })
        cur.execute("NOTIFY prediction_tick, %s", (json.dumps(payload),))
    conn.commit()


def main():
    conn=db.get_connection(); db_schema.migrate(conn)
    runtime_config.load_from_db(conn)
    listener=_listen_conn()
    monitors={}
    unavailable_machines=set()
    state_detectors={}
    table=db.get_table_name(); cols=db.get_db_columns(); last_seen=None
    print("Spindle Condition Monitoring worker started")
    while True:
        # Notifications are applied only at a tick boundary.
        if select.select([listener],[],[],0)[0]:
            listener.poll()
            while listener.notifies:
                note=listener.notifies.pop(0)
                if note.channel=="config_changed":
                    previous={key:runtime_config.get(key) for key in MONITOR_RESET_KEYS}
                    runtime_config.load_from_db(conn)
                    changed={key for key in MONITOR_RESET_KEYS if runtime_config.get(key)!=previous[key]}
                    if changed:
                        # Health scaling and buffer sizes cannot be mixed with
                        # state calculated under the previous policy.
                        monitors.clear()
                        unavailable_machines.clear()
                        print(f"[worker] Monitoring state reset after config change: {sorted(changed)}")
                elif note.channel=="model_changed":
                    predict_realtime.reload_model()
                    # A model/calibration change starts fresh state for every
                    # unit; carrying Kalman/trend state across models would
                    # blend two different health scales.
                    monitors.clear()
                    unavailable_machines.clear()
                elif note.channel=="env_changed":
                    print("Environment changed; graceful worker restart requested")
                    return 75
        if last_seen is None:
            state_history=db.fetch_recent_rows(conn,table,rows_per_machine=config.OPERATING_STATE_HISTORY_ROWS)
            for row in state_history:
                machine_id=db.row_machine_id(row,cols)
                detector=state_detectors.setdefault(machine_id,operating_state.OperatingStateDetector())
                historical=db.canonical_sensor_reading(row,cols)
                detector.observe_history(historical)
            for detector in state_detectors.values():
                detector.fit_history()
            warmup_rows=(
                config.WINDOW_SIZE
                + int(runtime_config.get("KALMAN_INIT_SAMPLES", config.KALMAN_INIT_SAMPLES))
                + 5
            )
            rows=db.fetch_recent_rows(conn,table,rows_per_machine=warmup_rows)
            calibrated=sum(detector.stop_threshold is not None for detector in state_detectors.values())
            print(f"[worker] Operating-state calibration: {calibrated}/{len(state_detectors)} machines have distinct motion regimes")
            print(f"[worker] Initial source sync: {len(rows)} recent rows across machines")
        else:
            rows=db.fetch_new_rows(conn,table,since=last_seen)
        for row in rows:
            machine_id=db.row_machine_id(row,cols)
            reading=db.canonical_sensor_reading(row,cols)
            detector=state_detectors.setdefault(machine_id,operating_state.OperatingStateDetector())
            state=detector.update(reading).to_dict()
            tick_ts=row[cols["timestamp"]]
            last_seen=(tick_ts,machine_id) if cols["machine_id"] else tick_ts
            result=None
            if state["state"] == "STOPPED":
                # Drop all rolling/Kalman/trend/debouncer state once. A future
                # STARTING transition creates a completely fresh monitor.
                if state["changed"]:
                    monitors.pop(machine_id,None)
            elif state["state"] != "SENSOR_FAULT" and not state["low_motion"]:
                monitor=monitors.get(machine_id)
                if monitor is None:
                    try:
                        monitor=_new_monitor(conn,machine_id); monitors[machine_id]=monitor
                        unavailable_machines.discard(machine_id)
                    except ValueError as exc:
                        if machine_id not in unavailable_machines:
                            print(f"[worker] Scoring disabled for {machine_id!r}: {exc}")
                            unavailable_machines.add(machine_id)
                        _persist_tick(conn,machine_id,reading,state,None,tick_ts)
                        continue
                numeric_reading={col:float(reading[col]) for col in config.RAW_SENSOR_COLS}
                result=monitor.update(numeric_reading,timestamp=tick_ts)
                # STARTING is an applicability gate even if feature/Kalman
                # warm-up happens to complete on the same tick.
                if state["state"] == "STARTING":
                    result=None
            _persist_tick(conn,machine_id,reading,state,result,tick_ts)
        time.sleep(float(runtime_config.get("WORKER_POLL_SECONDS", config.WORKER_POLL_SECONDS)))

if __name__=="__main__": sys.exit(main())
