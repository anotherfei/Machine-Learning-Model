"""Historical re-prediction using the exact production SpindleMonitor implementation."""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
import pandas as pd
import psycopg2.extras
import artifact_utils, config, db, maintenance, model_registry, operating_state, predict_realtime, runtime_config


STATUS_PATH=Path(config.ARTIFACTS_DIR)/"runtime"/"backfill_status.json"


def write_runtime_status(payload: dict) -> None:
    """Publish cross-process UI progress without depending on PostgreSQL."""
    STATUS_PATH.parent.mkdir(parents=True,exist_ok=True)
    temporary=STATUS_PATH.with_name(
        f".{STATUS_PATH.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(artifact_utils.to_json_safe(payload),allow_nan=False),
            encoding="utf-8",
        )
        os.replace(temporary,STATUS_PATH)
    finally:
        if temporary.exists():
            temporary.unlink()


def read_runtime_status() -> dict | None:
    try:
        value=json.loads(STATUS_PATH.read_text(encoding="utf-8"))
    except (FileNotFoundError,json.JSONDecodeError,OSError):
        return None
    return value if isinstance(value,dict) else None


class ModelVersionChanged(RuntimeError):
    """Raised when a pinned historical replay is superseded during execution."""


class CatchupContextChanged(RuntimeError):
    """Raised when mutable inference settings change during a stateful replay."""


def warmup_start(start):
    """Return the source-row warm-up boundary used by every replay caller."""
    return pd.Timestamp(start)-pd.Timedelta(minutes=max(
        int(runtime_config.get("TREND_LOOKBACK_MINUTES", config.TREND_LOOKBACK_MINUTES)),
        config.WINDOW_SIZE + int(runtime_config.get("KALMAN_INIT_SAMPLES", config.KALMAN_INIT_SAMPLES)),
    ))


def _flush_predictions(conn, records, skip_existing=False):
    """Persist a bounded replay batch while preserving reviewed evidence."""
    if not records:
        return 0, 0, 0
    conflict_clause=(
        """ON CONFLICT(machine_id,tick_timestamp,model_version) DO NOTHING"""
        if skip_existing else
        """ON CONFLICT(machine_id,tick_timestamp,model_version) DO UPDATE SET
                raw_reading=EXCLUDED.raw_reading,
                anomaly_score=EXCLUDED.anomaly_score,
                health_raw=EXCLUDED.health_raw,
                health_state=EXCLUDED.health_state,
                trend_slope_per_day=EXCLUDED.trend_slope_per_day,
                remaining_days=EXCLUDED.remaining_days,
                failure_probability=EXCLUDED.failure_probability,
                maintenance_level=EXCLUDED.maintenance_level,
                maintenance_reason=EXCLUDED.maintenance_reason,
                maintenance_trigger=EXCLUDED.maintenance_trigger,
                top_contributors=EXCLUDED.top_contributors,
                feature_vector=EXCLUDED.feature_vector,
                is_backfill=TRUE,
                alert_status=EXCLUDED.alert_status
              WHERE (spindle_predictions.alert_status IS NULL
                     OR spindle_predictions.alert_status='pending')
                AND spindle_predictions.near_miss_status IS NULL"""
    )
    with conn.cursor() as cur:
        returned = psycopg2.extras.execute_values(
            cur,
            """INSERT INTO spindle_predictions
              (machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,
               health_raw,health_state,trend_slope_per_day,remaining_days,
               failure_probability,maintenance_level,maintenance_reason,
               maintenance_trigger,top_contributors,feature_vector,is_backfill,
               alert_status)
              VALUES %s
              """+conflict_clause+"""
              RETURNING id""",
            records,
            template=(
                "(%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,"
                "%s::jsonb,%s::jsonb,%s,%s)"
            ),
            page_size=2000,
            fetch=True,
        )
        persisted = len(returned)
    skipped=len(records)-persisted
    return (
        persisted,
        0 if skip_existing else skipped,
        skipped if skip_existing else 0,
    )


def _existing_prediction_ticks(conn, machine_id, model_version, start, end):
    """Return same-model ticks already stored in one bounded replay chunk."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT tick_timestamp FROM spindle_predictions
               WHERE machine_id=%s AND model_version=%s
                 AND tick_timestamp>=%s AND tick_timestamp<=%s""",
            (machine_id,model_version,start,end),
        )
        return {pd.Timestamp(row[0]) for row in cur.fetchall()}


def run(
    conn,start,end,mode="repredict",machine_id=None,*,
    skip_existing=False,expected_model_version=None,progress_callback=None,
    bundle=None,return_runtime=False,continuity_guard=None,
    expected_source_rows=None,selected_machine_ids=None,
):
    start=pd.Timestamp(start); end=pd.Timestamp(end)
    if end <= start: raise ValueError("end must be after start")
    if mode=="skip":
        return {
            "mode":"skip","processed":0,"skipped_reviewed":0,
            "skipped_existing":0,"machine_ids":[],
            "start":start.isoformat(),"end":end.isoformat(),
        }
    if mode!="repredict": raise ValueError("mode must be repredict or skip")
    warmup=warmup_start(start)
    table=db.get_table_name(); c=db.get_db_columns()
    if machine_id and not c["machine_id"] and machine_id != db.DEFAULT_MACHINE_ID:
        return {
            "mode":"repredict","processed":0,"skipped_reviewed":0,
            "skipped_existing":0,"machine_ids":[],
            "start":start.isoformat(),"end":end.isoformat(),
        }
    selected_machines=(
        [machine_id]
        if machine_id else list(selected_machine_ids or db.fetch_machine_ids(conn,table,limit=10000))
    )
    bundle=bundle or predict_realtime.ModelBundle()
    if expected_model_version and bundle.version_id != expected_model_version:
        raise ModelVersionChanged(
            f"Historical replay expected model {expected_model_version}, "
            f"but loaded {bundle.version_id}."
        )
    pinned_version=expected_model_version or bundle.version_id
    written=0; machine_ids=set(); skipped_reviewed=0; skipped_existing=0
    runtime_monitors={};runtime_detectors={};runtime_levels={};runtime_snapshots={};last_seen=None
    replay_end=end+pd.Timedelta(microseconds=1)
    total_source_rows=0
    replay_started=time.monotonic()
    machine_count=max(1,len(selected_machines))

    def assert_continuity():
        if expected_model_version and model_registry.active_version()!=pinned_version:
            raise ModelVersionChanged(
                f"Active model changed from {pinned_version} during historical replay."
            )
        if continuity_guard is not None:
            continuity_guard()

    def report_progress(
        machine_number,machine_id,timestamp,status="running",*,phase="replay",message=None,
    ):
        if progress_callback is None:
            return
        if expected_source_rows:
            progress=max(0.0,min(1.0,total_source_rows/expected_source_rows))
        else:
            span=max((replay_end-warmup).total_seconds(),1e-9)
            position=(pd.Timestamp(timestamp)-warmup).total_seconds()/span
            position=max(0.0,min(1.0,float(position)))
            progress=max(0.0,min(1.0,((machine_number-1)+position)/machine_count))
        elapsed=max(time.monotonic()-replay_started,1e-9)
        eta=(elapsed*(1.0-progress)/progress) if 0.0<progress<1.0 else None
        displayed_total=expected_source_rows
        total_is_estimate=False
        if not displayed_total and total_source_rows and progress>0:
            displayed_total=max(total_source_rows,round(total_source_rows/progress))
            total_is_estimate=True
        progress_callback({
            "status":status,
            "phase":phase,
            "message":message,
            "progress":progress,
            "machine_id":machine_id,
            "machine_number":machine_number,
            "machine_count":len(selected_machines),
            "source_rows":total_source_rows,
            "source_rows_total":displayed_total,
            "source_rows_total_estimated":total_is_estimate,
            "predictions_written":written,
            "skipped_existing":skipped_existing,
            "skipped_reviewed":skipped_reviewed,
            "rows_per_second":total_source_rows/elapsed if total_source_rows else 0.0,
            "eta_seconds":eta,
            "model_version":pinned_version,
            "range_start":start.isoformat(),
            "range_end":end.isoformat(),
        })

    report_progress(1,selected_machines[0] if selected_machines else None,warmup)
    for index,row_machine_id in enumerate(selected_machines,1):
        assert_continuity()
        report_progress(
            index,row_machine_id,warmup,
            phase="motion_profile",
            message=f"Learning operating-state motion regimes for {row_machine_id}.",
        )
        detector=operating_state.OperatingStateDetector()
        runtime_detectors[row_machine_id]=detector
        profile_rows=db.fetch_rows_before(
            conn,table,replay_end.to_pydatetime(),
            config.OPERATING_STATE_HISTORY_ROWS,row_machine_id,
        )
        for profile_row in profile_rows:
            detector.observe_history(db.canonical_sensor_reading(profile_row,c))
        detector.fit_history()
        report_progress(
            index,row_machine_id,warmup,
            phase="replay",
            message=f"Replaying historical readings for {row_machine_id}.",
        )
        monitor=None; previous_level=None; state_changed_at=None
        for rows in db.iter_row_chunks_between(
            table,warmup.to_pydatetime(),replay_end.to_pydatetime(),
            machine_id=row_machine_id,
            chunk_rows=predict_realtime.INFERENCE_BATCH_ROWS,
        ):
            assert_continuity()
            records=[]
            pending=[]
            existing_ticks=(
                _existing_prediction_ticks(
                    conn,row_machine_id,pinned_version,
                    rows[0][c["timestamp"]],rows[-1][c["timestamp"]],
                )
                if skip_existing and pd.Timestamp(rows[-1][c["timestamp"]]) >= start
                else set()
            )

            def score_pending():
                nonlocal previous_level,skipped_existing
                if not pending:
                    return
                results=monitor.update_many(
                    [item[1] for item in pending],
                    [item[0] for item in pending],
                )
                for (tick,numeric_reading,state_name),result in zip(pending,results):
                    if result is None or state_name == "STARTING":
                        continue
                    current_level=result["maintenance"]["level"]
                    emit_alert=maintenance.is_alert_escalation(previous_level,current_level)
                    previous_level=current_level
                    if tick < start:
                        continue
                    if tick in existing_ticks:
                        existing_ticks.discard(tick)
                        skipped_existing+=1
                        continue
                    result=artifact_utils.to_json_safe(result)
                    rec=result["maintenance"]
                    raw={key:result[key] for key in config.RAW_SENSOR_COLS}
                    records.append((
                        row_machine_id,tick,result["model_version"],json.dumps(raw,allow_nan=False),
                        result["anomaly_score"],result["health_raw"],result["health_state"],
                        result["trend_slope_per_day"],result["remaining_days"],
                        json.dumps(result["failure_probability"],allow_nan=False),
                        rec["level"],rec["reason"],rec.get("trigger","none"),
                        json.dumps(result.get("top_contributors"),allow_nan=False),
                        json.dumps(result.get("feature_vector"),allow_nan=False),True,
                        "pending" if emit_alert else None,
                    ))
                pending.clear()

            for row in rows:
                total_source_rows+=1; machine_ids.add(row_machine_id)
                tick=pd.Timestamp(row[c["timestamp"]])
                cursor=(tick.to_pydatetime(),row_machine_id) if c["machine_id"] else tick.to_pydatetime()
                if last_seen is None or cursor>last_seen:
                    last_seen=cursor
                reading=db.canonical_sensor_reading(row,c)
                state=detector.update(reading)
                if state_changed_at is None or state.changed:
                    state_changed_at=tick.to_pydatetime()
                runtime_snapshots[row_machine_id]={
                    "state":state.to_dict(),
                    "tick_timestamp":tick.to_pydatetime(),
                    "state_changed_at":state_changed_at,
                }
                if state.state == "STOPPED":
                    if state.changed:
                        score_pending()
                        monitor=None; previous_level=None
                    continue
                if state.state == "SENSOR_FAULT" or state.low_motion:
                    continue
                if monitor is None:
                    monitor=bundle.create_monitor(row_machine_id)
                numeric_reading={col:float(reading[col]) for col in config.RAW_SENSOR_COLS}
                pending.append((tick,numeric_reading,state.state))
            score_pending()
            # A raw source should normally have one row per machine/timestamp,
            # but tolerate duplicates without asking PostgreSQL to update the
            # same conflict target twice in one batched INSERT.
            records=list({(record[0],record[1],record[2]):record for record in records}.values())
            persisted,reviewed,existing=_flush_predictions(
                conn,records,skip_existing=skip_existing,
            )
            written+=persisted; skipped_reviewed+=reviewed; skipped_existing+=existing
            report_progress(
                index,row_machine_id,rows[-1][c["timestamp"]],
                phase="replay",
                message=f"Replaying historical readings for {row_machine_id}.",
            )
            conn.commit()
        if monitor is not None:
            runtime_monitors[row_machine_id]=monitor
        if previous_level is not None:
            runtime_levels[row_machine_id]=previous_level
        report_progress(
            index,row_machine_id,replay_end,
            phase="machine_complete",
            message=f"Completed historical replay for {row_machine_id}.",
        )
        conn.commit()
    assert_continuity()
    report_progress(
        machine_count,selected_machines[-1] if selected_machines else None,
        replay_end,status="completed",phase="handoff",
        message="Historical replay complete; preparing realtime handoff.",
    )
    conn.commit()
    summary={
        "mode":"repredict","processed":written,"skipped_reviewed":skipped_reviewed,
        "skipped_existing":skipped_existing,"model_version":pinned_version,
        "source_rows":total_source_rows,"source_rows_total":expected_source_rows,
        "machine_ids":sorted(machine_ids),"start":start.isoformat(),"end":end.isoformat(),
    }
    if not return_runtime:
        return summary
    return summary,{
        "bundle":bundle,"monitors":runtime_monitors,
        "state_detectors":runtime_detectors,
        "maintenance_levels":runtime_levels,"last_seen":last_seen,
        "operating_snapshots":runtime_snapshots,
    }
