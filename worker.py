"""Long-running production worker: polling, prediction persistence, alerts, hot reload."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import json
import select
import sys
import time
import uuid
import psycopg2.extras

import artifact_utils
import backfill
import config
import db
import db_schema
import env_manager
import maintenance
import model_registry
import operating_state
import predict_realtime
import runtime_config

MONITOR_RESET_KEYS = {
    "HEALTH_SENSITIVITY_STD",
    "KALMAN_INIT_SAMPLES",
    "TREND_LOOKBACK_MINUTES",
}
MAINTENANCE_STATE_RESET_KEYS = {
    "FAILURE_HEALTH_THRESHOLD",
    "MAINTENANCE_HEALTH_INSPECT",
    "MAINTENANCE_PROB_PLAN",
    "MAINTENANCE_PROB_URGENT",
    "MAINTENANCE_HORIZON_DAYS",
    "MAINTENANCE_URGENT_HORIZON_DAYS",
    "MAINTENANCE_WARN_CONFIRM_MINUTES",
    "MAINTENANCE_CRITICAL_CONFIRM_MINUTES",
    "MAINTENANCE_RECOVERY_MINUTES",
    "TREND_MIN_POINTS",
    "TREND_SLOPE_Z_THRESHOLD",
    "TREND_SETTLE_TICKS",
}
CONFIG_CHANGE_KEYS = MONITOR_RESET_KEYS | MAINTENANCE_STATE_RESET_KEYS
CATCHUP_CONTEXT_KEYS = set(runtime_config.THRESHOLD_KEYS)


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the production inference worker")
    parser.add_argument(
        "--catch-up-days",
        type=int,
        default=0,
        help="replay this many source days, then continue live with the warmed runtime state",
    )
    parser.add_argument(
        "--schema-ready",
        action="store_true",
        help="skip migration because the supervised API already completed it",
    )
    args = parser.parse_args()
    if not 0 <= args.catch_up_days <= 3650:
        parser.error("--catch-up-days must be between 0 and 3650")
    return args


def _write_backfill_status(_conn, payload: dict) -> None:
    document={
        **payload,
        "updated_at":datetime.now(timezone.utc).isoformat(),
    }
    # This local atomic document is authoritative for live progress. Keeping
    # status writes off the inference connection prevents observability from
    # blocking behind a saturated PostgreSQL source.
    backfill.write_runtime_status(document)


def _persist_catchup_operating_states(conn, snapshots: dict) -> None:
    """Publish replay-end machine state without creating duplicate predictions."""
    with conn.cursor() as cur:
        for machine_id,snapshot in snapshots.items():
            state=snapshot["state"]
            cur.execute(
                """INSERT INTO state(namespace,key,value,updated_at)
                   VALUES(
                     'machine',%s,
                     jsonb_build_object(
                       'operating_state',%s,'reason',%s,'confidence',%s,
                       'activity_score',%s,'stop_threshold',%s,'run_threshold',%s,
                       'tick_timestamp',%s,'state_changed_at',%s
                     ),now()
                   )
                   ON CONFLICT(namespace,key) DO UPDATE SET
                     value=EXCLUDED.value,updated_at=now()""",
                (
                    machine_id,state["state"],state["reason"],state["confidence"],
                    state["activity_score"],state["stop_threshold"],state["run_threshold"],
                    snapshot["tick_timestamp"],snapshot["state_changed_at"],
                ),
            )


def _catchup_context_revision(conn):
    """Return a non-secret fingerprint for settings that affect live continuity."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT max(updated_at) FILTER (
                        WHERE namespace='config' AND key=ANY(%s)
                      ),
                      max(updated_at) FILTER (WHERE namespace='env_audit')
               FROM state
               WHERE namespace IN ('config','env_audit')""",
            (sorted(CATCHUP_CONTEXT_KEYS),),
        )
        config_updated_at,env_updated_at=cur.fetchone()
    env_mtime=(
        env_manager.ENV_PATH.stat().st_mtime_ns
        if env_manager.ENV_PATH.exists() else None
    )
    return config_updated_at,env_updated_at,env_mtime


def _run_sequential_catchup(conn, days: int):
    """Replay a fixed history and return its exact warmed runtime state."""
    if days <= 0:
        return 0,None
    launch_status=backfill.read_runtime_status() or {}
    reuse_launch=(
        launch_status.get("status") in {"launching","connecting"}
        and int(launch_status.get("days",-1))==days
    )
    run_id=str(launch_status.get("run_id")) if reuse_launch else uuid.uuid4().hex
    started_at=(
        str(launch_status.get("started_at"))
        if reuse_launch and launch_status.get("started_at")
        else datetime.now(timezone.utc).isoformat()
    )
    base={
        "run_id":run_id,"status":"preparing","progress":0.0,"days":days,
        "started_at":started_at,
        "message":"Discovering machines and the newest source watermark.",
    }
    _write_backfill_status(conn,base);conn.commit()
    print(
        "[catch-up] Preparing historical replay. Progress is available in the web console.",
        flush=True,
    )
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT set_config('statement_timeout',%s,false)",
                (f"{int(config.TRAINING_DB_STATEMENT_TIMEOUT_MS)}ms",),
            )
        conn.commit()
        # Re-read database policy only after the API-visible preparation lock
        # exists, then fingerprint that exact context for the whole replay.
        runtime_config.load_from_db(conn)
        context_revision=_catchup_context_revision(conn)
        table = db.get_table_name()
        columns = db.get_db_columns()
        machine_ids=db.fetch_machine_ids(conn,table,limit=10000)
        newest_rows=[]
        for index,machine_id in enumerate(machine_ids,1):
            _write_backfill_status(conn,{
                **base,
                "message":f"Finding the newest source row for {machine_id} ({index}/{len(machine_ids)}).",
            })
            row=db.fetch_latest_row(conn,table,machine_id)
            if row is not None:
                newest_rows.append(row)
        if not newest_rows:
            raise ValueError("The configured PostgreSQL source contains no sensor rows to backfill.")
        source_end = max(row[columns["timestamp"]] for row in newest_rows)
        source_start = source_end - timedelta(days=days)
        version_id=model_registry.active_version()
        if not version_id:
            raise ValueError("No active model is available for historical backfill.")
        bundle=predict_realtime.ModelBundle()
        replay_end=source_end+timedelta(microseconds=1)
        replay_start=backfill.warmup_start(source_start).to_pydatetime()
        _write_backfill_status(conn,{
            **base,
            "message":"Counting historical source rows for progress reporting.",
        })
        try:
            expected_source_rows=db.count_rows_between(
                conn,table,replay_start,replay_end,machine_ids,
            )
            conn.commit()
        except Exception as count_error:
            conn.rollback()
            expected_source_rows=None
            print(
                f"[catch-up] Exact source-row count unavailable ({count_error}); "
                "progress will use a live estimate.",
                flush=True,
            )
    except Exception as exc:
        _write_backfill_status(conn,{
            **base,"status":"failed","error":str(exc),
            "finished_at":datetime.now(timezone.utc).isoformat(),
        });conn.commit()
        print(
            f"[catch-up] Could not prepare historical replay: {exc}",
            file=sys.stderr,flush=True,
        )
        return 1,None
    base={
        **base,"status":"running","model_version":version_id,
        "range_start":str(source_start),"range_end":str(source_end),
        "source_rows_total":expected_source_rows,
        "message":"Replaying historical readings through the production pipeline.",
    }
    _write_backfill_status(conn,base);conn.commit()
    print(
        f"[catch-up] Historical replay started for model {version_id}: "
        f"{source_start} through {source_end}. Progress is available in the web console.",
        flush=True,
    )
    last_published=0.0
    latest_progress={}

    def publish(update):
        nonlocal last_published,latest_progress
        if update.get("status")=="completed":
            update={**update,"status":"draining","progress":0.99}
        phase_changed=update.get("phase")!=latest_progress.get("phase")
        latest_progress=dict(update)
        now=time.monotonic()
        if (
            update.get("status")=="running"
            and not phase_changed
            and now-last_published<1.0
        ):
            return
        last_published=now
        _write_backfill_status(conn,{**base,**update})

    def guard_context():
        if _catchup_context_revision(conn)!=context_revision:
            raise backfill.CatchupContextChanged(
                "Inference policy or environment settings changed during historical replay."
            )

    try:
        summary,runtime = backfill.run(
            conn,source_start,source_end,mode="repredict",
            skip_existing=True,expected_model_version=version_id,
            progress_callback=publish,bundle=bundle,return_runtime=True,
            continuity_guard=guard_context,
            expected_source_rows=expected_source_rows,
            selected_machine_ids=machine_ids,
        )
    except (backfill.ModelVersionChanged,backfill.CatchupContextChanged) as exc:
        _write_backfill_status(conn,{
            **base,"status":"restarting","message":str(exc),
            "updated_at":datetime.now(timezone.utc).isoformat(),
        });conn.commit()
        print(f"[catch-up] {exc} Restarting with a consistent runtime context.")
        return 76,None
    except Exception as exc:
        _write_backfill_status(conn,{
            **base,"status":"failed","error":str(exc),
            "finished_at":datetime.now(timezone.utc).isoformat(),
        });conn.commit()
        print(f"[catch-up] Historical replay failed: {exc}",file=sys.stderr)
        return 1,None
    handoff={
        **base,**latest_progress,**summary,
        "status":"draining","progress":0.99,
        "predictions_written":summary["processed"],
        "message":"Historical range complete; draining rows that arrived during catch-up.",
    }
    _write_backfill_status(conn,handoff);conn.commit()
    runtime["catchup_status"]=handoff
    runtime["context_revision"]=context_revision
    print(
        f"[catch-up] Complete: {summary['processed']:,} new predictions, "
        f"{summary['skipped_existing']:,} existing same-model predictions preserved; "
        "continuing live with the warmed state."
    )
    return 0,runtime


def _latest_maintenance_levels(conn):
    """Seed transition detection from persisted predictions after a restart."""
    with conn.cursor() as cur:
        cur.execute(
            """SELECT DISTINCT ON (machine_id) machine_id, maintenance_level
               FROM spindle_predictions
               ORDER BY machine_id, tick_timestamp DESC, id DESC"""
        )
        return {machine_id: level for machine_id, level in cur.fetchall()}


def _listen_conn():
    c = db.get_connection(); c.set_isolation_level(0)
    with c.cursor() as cur:
        cur.execute(
            "LISTEN model_changed; LISTEN config_changed; LISTEN env_changed; "
            "LISTEN operating_override_changed;"
        )
    return c


def _persist_tick(
    conn,machine_id,reading,state,result,tick_ts,emit_alert=False,*,
    commit=True,notify=True,
):
    # NumPy's round() may return np.float64 rather than a native Python float.
    # psycopg2 can render that object as the SQL text ``np.float64(...)``, which
    # PostgreSQL interprets as a nonexistent schema/function. Normalize the
    # entire inference result once at the persistence boundary, including
    # nested probability, attribution, and feature-vector values.
    result = artifact_utils.to_json_safe(result) if result is not None else None
    raw = {
        key:artifact_utils.finite_float(value)
        for key, value in reading.items()
    }
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO state AS existing(namespace,key,value,updated_at)
               VALUES(
                 'machine',%s,
                 jsonb_build_object(
                   'operating_state',%s,'reason',%s,'confidence',%s,
                   'activity_score',%s,'stop_threshold',%s,'run_threshold',%s,
                   'low_motion',%s,'state_source',%s,
                   'detected_operating_state',%s,'detected_reason',%s,'detected_confidence',%s,
                   'override_set_by',%s,'override_set_at',%s,'override_expires_at',%s,'override_note',%s,
                   'tick_timestamp',%s,'state_changed_at',%s
                 ),now()
               )
               ON CONFLICT(namespace,key) DO UPDATE SET
                 value=EXCLUDED.value || jsonb_build_object(
                   'state_changed_at',CASE
                     WHEN existing.value->>'operating_state'
                          IS DISTINCT FROM EXCLUDED.value->>'operating_state'
                     THEN EXCLUDED.value->'tick_timestamp'
                     ELSE existing.value->'state_changed_at'
                   END
                 ),
                 updated_at=now()""",
            (
                machine_id,state["state"],state["reason"],state["confidence"],state["activity_score"],
                state["stop_threshold"],state["run_threshold"],state["low_motion"],state.get("state_source","automatic"),
                state.get("detected_state",state["state"]),state.get("detected_reason",state["reason"]),
                state.get("detected_confidence",state["confidence"]),state.get("override_set_by"),
                state.get("override_set_at"),state.get("override_expires_at"),state.get("override_note"),
                tick_ts,tick_ts,
            ),
        )

        if result is not None:
            rec = result["maintenance"]
            cur.execute("""INSERT INTO spindle_predictions
                (machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,health_raw,health_state,trend_slope_per_day,remaining_days,
                 failure_probability,maintenance_level,maintenance_reason,maintenance_trigger,
                 top_contributors,feature_vector,alert_status)
                 VALUES(%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s::jsonb,%s,%s,%s,%s::jsonb,%s::jsonb,%s)
                 ON CONFLICT(machine_id,tick_timestamp,model_version) DO NOTHING RETURNING id""",
                (machine_id,tick_ts,result["model_version"],json.dumps(raw,allow_nan=False),result["anomaly_score"],result["health_raw"],result["health_state"],
                 result["trend_slope_per_day"],result["remaining_days"],json.dumps(result["failure_probability"],allow_nan=False),rec["level"],rec["reason"],rec.get("trigger","none"),
                 json.dumps(result.get("top_contributors"),allow_nan=False),
                 json.dumps(result.get("feature_vector"),allow_nan=False),
                 "pending" if emit_alert else None))
            cur.fetchone()

        payload={
            "machine_id":machine_id,
            "timestamp":str(tick_ts),
            "prediction_timestamp":str(tick_ts) if result is not None else None,
            "prediction_lag_seconds":0.0 if result is not None else None,
            "worker_poll_seconds":runtime_config.get("WORKER_POLL_SECONDS", config.WORKER_POLL_SECONDS),
            "operating_state":state["state"],
            "operating_state_reason":state["reason"],
            "operating_state_confidence":state["confidence"],
            "operating_state_source":state.get("state_source","automatic"),
            "detected_operating_state":state.get("detected_state",state["state"]),
            "detected_operating_state_reason":state.get("detected_reason",state["reason"]),
            "operating_override":({
                "state":state["state"],"set_by":state.get("override_set_by"),
                "set_at":state.get("override_set_at"),"expires_at":state.get("override_expires_at"),
                "note":state.get("override_note"),
            } if state.get("state_source")=="operator" else None),
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
        if notify:
            cur.execute("NOTIFY prediction_tick, %s", (json.dumps(payload),))
    if commit:
        conn.commit()


def main():
    args = _parse_args()
    if args.catch_up_days:
        launch_status=backfill.read_runtime_status() or {}
        reuse_launch=(
            launch_status.get("status")=="launching"
            and int(launch_status.get("days",-1))==args.catch_up_days
        )
        if not reuse_launch:
            launch_status={
                "run_id":uuid.uuid4().hex,
                "started_at":datetime.now(timezone.utc).isoformat(),
            }
        _write_backfill_status(None,{
            **launch_status,
            "status":"connecting",
            "progress":0.0,
            "days":args.catch_up_days,
            "message":"Connecting the production worker to PostgreSQL.",
        })
    try:
        conn=db.get_connection()
    except Exception as exc:
        if args.catch_up_days:
            _write_backfill_status(None,{
                **(backfill.read_runtime_status() or {}),
                "status":"failed","error":f"PostgreSQL connection failed: {exc}",
                "finished_at":datetime.now(timezone.utc).isoformat(),
            })
        print(f"[worker] PostgreSQL connection failed: {exc}",file=sys.stderr,flush=True)
        return 1
    if not args.schema_ready:
        db_schema.migrate(conn)
    catchup_runtime=None
    if args.catch_up_days:
        try:
            code,catchup_runtime=_run_sequential_catchup(conn,args.catch_up_days)
            if code:
                return code
        except Exception as exc:
            _write_backfill_status(conn,{
                **(backfill.read_runtime_status() or {}),
                "status":"failed","progress":0.0,
                "error":str(exc),"days":args.catch_up_days,
                "finished_at":datetime.now(timezone.utc).isoformat(),
            });conn.commit()
            print(
                f"[catch-up] Could not start historical replay: {exc}",
                file=sys.stderr,flush=True,
            )
            return 1
    else:
        runtime_config.load_from_db(conn)
    bundle=(catchup_runtime or {}).get("bundle") or predict_realtime.ModelBundle()
    listener=_listen_conn()
    context_changed=(
        catchup_runtime is not None
        and _catchup_context_revision(conn)!=catchup_runtime.get("context_revision")
    )
    if catchup_runtime is not None and (
        model_registry.active_version()!=bundle.version_id or context_changed
    ):
        _write_backfill_status(conn,{
            **catchup_runtime.get("catchup_status",{}),"status":"restarting",
            "message":"Model or inference settings changed before realtime handoff; restarting catch-up.",
        });conn.commit()
        return 76
    if catchup_runtime is not None:
        try:
            _persist_catchup_operating_states(
                conn,catchup_runtime.get("operating_snapshots",{}),
            )
            conn.commit()
        except Exception as exc:
            _write_backfill_status(conn,{
                **catchup_runtime.get("catchup_status",{}),"status":"failed",
                "error":f"Could not publish replay-end machine state: {exc}",
                "finished_at":datetime.now(timezone.utc).isoformat(),
            });conn.commit()
            print(f"[catch-up] Could not publish replay-end machine state: {exc}",file=sys.stderr)
            return 1
    monitors=(catchup_runtime or {}).get("monitors",{})
    unavailable_machines=set()
    state_detectors=(catchup_runtime or {}).get("state_detectors",{})
    operator_overrides=operating_state.load_active_overrides(conn)
    effective_states={}
    maintenance_levels=(
        catchup_runtime["maintenance_levels"]
        if catchup_runtime is not None else _latest_maintenance_levels(conn)
    )
    table=db.get_table_name(); cols=db.get_db_columns()
    last_seen=(catchup_runtime or {}).get("last_seen")
    catchup_active=catchup_runtime is not None
    catchup_status=(catchup_runtime or {}).get("catchup_status",{})
    print("Spindle Condition Monitoring worker started")
    while True:
        initial_sync = last_seen is None
        # Notifications are applied only at a tick boundary.
        if select.select([listener],[],[],0)[0]:
            listener.poll()
            while listener.notifies:
                note=listener.notifies.pop(0)
                if note.channel=="config_changed":
                    previous={key:runtime_config.get(key) for key in CONFIG_CHANGE_KEYS}
                    runtime_config.load_from_db(conn)
                    changed={key for key in CONFIG_CHANGE_KEYS if runtime_config.get(key)!=previous[key]}
                    rebuild=changed & MONITOR_RESET_KEYS
                    if rebuild:
                        # Health scaling and buffer sizes cannot be mixed with
                        # state calculated under the previous policy.
                        monitors.clear()
                        unavailable_machines.clear()
                        print(f"[worker] Monitoring state reset after config change: {sorted(rebuild)}")
                    elif changed & MAINTENANCE_STATE_RESET_KEYS:
                        # Keep feature/Kalman/trend history, but never carry a
                        # partially confirmed alert across a policy change.
                        for monitor in monitors.values():
                            monitor.maintenance=maintenance.MaintenanceDebouncer()
                        print(f"[worker] Maintenance status reset after config change: {sorted(changed)}")
                elif note.channel=="model_changed":
                    if catchup_active:
                        _write_backfill_status(conn,{
                            **catchup_status,"status":"restarting",
                            "message":"Active model changed before realtime handoff; restarting catch-up.",
                        });conn.commit()
                        return 76
                    bundle=predict_realtime.ModelBundle()
                    # A model/calibration change starts fresh state for every
                    # unit; carrying Kalman/trend state across models would
                    # blend two different health scales.
                    monitors.clear()
                    unavailable_machines.clear()
                elif note.channel=="env_changed":
                    print("Environment changed; graceful worker restart requested")
                    return 75
                elif note.channel=="operating_override_changed":
                    selected_machine=note.payload.strip()
                    active=operating_state.load_override(conn,selected_machine)
                    if active is None:
                        operator_overrides.pop(selected_machine,None)
                        print(f"[worker] Operator motion confirmation cleared for {selected_machine}")
                    else:
                        operator_overrides[selected_machine]=active
                        print(
                            f"[worker] Operator confirmed {selected_machine} {active['state']} "
                            f"until {active['expires_at']}"
                        )
        if last_seen is None:
            profile_started=time.monotonic()
            print(
                f"[worker] Loading the newest {config.OPERATING_STATE_HISTORY_ROWS} "
                "motion-profile rows per machine..."
            )
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
            print(
                f"[worker] Operating-state calibration: {calibrated}/{len(state_detectors)} "
                f"machines have distinct motion regimes ({time.monotonic()-profile_started:.1f}s)"
            )
            print(f"[worker] Initial source sync: {len(rows)} recent rows across machines")
        else:
            rows=db.fetch_new_rows(conn,table,since=last_seen)
        for row in rows:
            machine_id=db.row_machine_id(row,cols)
            reading=db.canonical_sensor_reading(row,cols)
            detector=state_detectors.setdefault(machine_id,operating_state.OperatingStateDetector())
            tick_ts=row[cols["timestamp"]]
            detected_state=detector.update(reading).to_dict()
            state=operating_state.apply_operator_override(
                detected_state,operator_overrides.get(machine_id),tick_ts
            )
            previous_effective=effective_states.get(machine_id)
            state["changed"]=(
                detected_state["changed"] if previous_effective is None
                else previous_effective != state["state"]
            )
            effective_states[machine_id]=state["state"]
            last_seen=(tick_ts,machine_id) if cols["machine_id"] else tick_ts
            result=None
            if state["state"] == "STOPPED":
                # Drop all rolling/Kalman/trend/debouncer state once. A future
                # STARTING transition creates a completely fresh monitor.
                if state["changed"]:
                    monitors.pop(machine_id,None)
                    maintenance_levels.pop(machine_id,None)
            elif state["state"] != "SENSOR_FAULT" and not state["low_motion"]:
                monitor=monitors.get(machine_id)
                if monitor is None:
                    try:
                        monitor=bundle.create_monitor(machine_id)
                        monitors[machine_id]=monitor
                        unavailable_machines.discard(machine_id)
                    except ValueError as exc:
                        if machine_id not in unavailable_machines:
                            print(f"[worker] Scoring disabled for {machine_id!r}: {exc}")
                            unavailable_machines.add(machine_id)
                        _persist_tick(
                            conn,machine_id,reading,state,None,tick_ts,
                            commit=not catchup_active,notify=not catchup_active,
                        )
                        continue
                numeric_reading={col:float(reading[col]) for col in config.RAW_SENSOR_COLS}
                result=monitor.update(numeric_reading,timestamp=tick_ts)
                # STARTING is an applicability gate even if feature/Kalman
                # warm-up happens to complete on the same tick.
                if state["state"] == "STARTING":
                    result=None
            emit_alert=False
            if result is not None:
                current_level=result["maintenance"]["level"]
                emit_alert=(
                    not initial_sync
                    and maintenance.is_alert_escalation(
                        maintenance_levels.get(machine_id), current_level
                    )
                )
                maintenance_levels[machine_id]=current_level
            _persist_tick(
                conn,machine_id,reading,state,result,tick_ts,
                emit_alert=emit_alert,
                commit=not catchup_active,notify=not catchup_active,
            )
        if catchup_active and rows:
            conn.commit()
            catchup_status={
                **catchup_status,"status":"draining","progress":0.99,
                "source_rows":int(catchup_status.get("source_rows",0))+len(rows),
                "message":"Historical range complete; draining rows that arrived during catch-up.",
            }
            _write_backfill_status(conn,catchup_status);conn.commit()
        # Immediately ask for the next page after processing rows. This drains
        # restart/backlog data without inserting an idle delay between batches.
        # Polling sleep applies only once the worker has caught up completely.
        if rows:
            continue
        if catchup_active:
            catchup_active=False
            catchup_status={
                **catchup_status,"status":"completed","progress":1.0,
                "finished_at":datetime.now(timezone.utc).isoformat(),
                "message":"Sequential catch-up reached the live source watermark.",
            }
            _write_backfill_status(conn,catchup_status);conn.commit()
            print("[catch-up] Realtime source reached; normal live polling is now active.")
        time.sleep(float(runtime_config.get("WORKER_POLL_SECONDS", config.WORKER_POLL_SECONDS)))

if __name__=="__main__": sys.exit(main())
