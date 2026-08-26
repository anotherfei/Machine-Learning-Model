"""PostgreSQL-backed coordinator for reusable historical accuracy simulations.

One compact ML.simulation_runs table stores reusable drafts, queued work,
progress, event-level evidence, and completed reports. Historical replay still
reads the plant sensor source without writing predictions or alerts.
"""
from __future__ import annotations

import datetime as dt
import json
import time

import psycopg2.errors
import psycopg2.extras

import artifact_utils
import db
import runtime_config
import simulation_service as replay


EXPECTED_STATUSES = replay.EXPECTED_STATUSES
MAX_CASES = replay.MAX_CASES
LEAD_WINDOW_DAYS = replay.LEAD_WINDOW_DAYS
MAX_EVENT_WINDOW_DAYS = replay.MAX_EVENT_WINDOW_DAYS
_DATE_FIELDS = ("created_at", "updated_at", "started_at", "finished_at")


def _json_value(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def _public_run(row: dict, *, include_cases: bool = True) -> dict:
    item = dict(row)
    for key in ("cases", "policy_snapshot", "summary"):
        item[key] = _json_value(item.get(key))
    for key in _DATE_FIELDS:
        value = item.get(key)
        if isinstance(value, (dt.datetime, dt.date)):
            item[key] = value.isoformat()
    cases = item.get("cases") or []
    item["case_count"] = len(cases)
    progress_items = [case.get("progress") for case in cases if case.get("progress")]
    if progress_items:
        item["progress"] = next(
            (value for value in reversed(progress_items) if value.get("status") != "completed"),
            progress_items[-1],
        )
    if not include_cases:
        item.pop("cases", None)
    return artifact_utils.to_json_safe(item)


def _saved_cases(run_id: int, cases: list[dict], *, results: bool) -> list[dict]:
    saved = []
    for position, source in enumerate(cases, start=1):
        item = {
            "id": run_id * 1000 + position,
            "run_id": run_id,
            "position": position,
            "machine_id": str(source.get("machine_id") or ""),
            "description": str(source.get("description") or ""),
            "expected_status": str(source.get("expected_status") or "OK").upper(),
            "start": source.get("start").isoformat() if isinstance(source.get("start"), dt.datetime) else str(source.get("start") or ""),
            "end": source.get("end").isoformat() if isinstance(source.get("end"), dt.datetime) else str(source.get("end") or ""),
        }
        if results:
            item["window_start"] = (
                source["start"] - dt.timedelta(days=LEAD_WINDOW_DAYS)
            ).isoformat()
            item["result"] = None
            item["error"] = None
            item["progress"] = None
        saved.append(item)
    return saved


def enqueue(name: str, model_version: str, cases: list[dict], created_by: str) -> dict:
    conn = db.get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT 1 FROM simulation_runs WHERE status IN ('queued','running') FOR UPDATE"
            )
            if cur.fetchone():
                raise ValueError("Another historical simulation is already queued or running")
            cur.execute(
                """INSERT INTO simulation_runs
                   (name,model_version,status,created_by,cases,policy_snapshot,total_cases)
                   VALUES(%s,%s,'queued',%s,'[]'::jsonb,%s,%s)
                   RETURNING *""",
                (
                    name,
                    model_version,
                    created_by,
                    psycopg2.extras.Json(artifact_utils.to_json_safe(replay.policy_snapshot())),
                    len(cases),
                ),
            )
            run = dict(cur.fetchone())
            run["cases"] = _saved_cases(int(run["id"]), cases, results=True)
            cur.execute(
                "UPDATE simulation_runs SET cases=%s,updated_at=now() WHERE id=%s RETURNING *",
                (psycopg2.extras.Json(run["cases"]), run["id"]),
            )
            created = dict(cur.fetchone())
        conn.commit()
        return _public_run(created)
    except psycopg2.errors.UniqueViolation as exc:
        conn.rollback()
        raise ValueError("Another historical simulation is already queued or running") from exc
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def save_draft(name: str, cases: list[dict], created_by: str, draft_id: int | None = None) -> dict:
    if not 1 <= len(cases) <= MAX_CASES:
        raise ValueError(f"A reusable list must contain 1 to {MAX_CASES} events")
    conn = db.get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if draft_id is None:
                cur.execute(
                    """INSERT INTO simulation_runs(name,status,created_by,cases,total_cases)
                       VALUES(%s,'draft',%s,'[]'::jsonb,%s) RETURNING *""",
                    (name, created_by, len(cases)),
                )
                draft = dict(cur.fetchone())
            else:
                cur.execute(
                    "SELECT * FROM simulation_runs WHERE id=%s AND status='draft' FOR UPDATE",
                    (draft_id,),
                )
                row = cur.fetchone()
                if not row:
                    raise ValueError("Reusable simulation list not found")
                draft = dict(row)
                cur.execute(
                    "UPDATE simulation_runs SET name=%s,total_cases=%s,updated_at=now() WHERE id=%s",
                    (name, len(cases), draft_id),
                )
            saved = _saved_cases(int(draft["id"]), cases, results=False)
            cur.execute(
                "UPDATE simulation_runs SET cases=%s,updated_at=now() WHERE id=%s RETURNING *",
                (psycopg2.extras.Json(saved), draft["id"]),
            )
            result = dict(cur.fetchone())
        conn.commit()
        return _public_run(result)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_run(run_id: int) -> dict | None:
    conn = db.get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM simulation_runs WHERE id=%s", (run_id,))
            row = cur.fetchone()
        return _public_run(dict(row)) if row else None
    finally:
        conn.close()


def list_runs(limit: int = 25, *, drafts: bool = False) -> list[dict]:
    conn = db.get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM simulation_runs
                   WHERE status {operator} 'draft'
                   ORDER BY updated_at DESC,id DESC LIMIT %s""".format(
                    operator="=" if drafts else "<>"
                ),
                (max(1, min(int(limit), 100)),),
            )
            rows = cur.fetchall()
        return [_public_run(dict(row), include_cases=drafts) for row in rows]
    finally:
        conn.close()


def set_validation_suite(draft_id: int, enabled: bool) -> dict:
    """Select at most one reusable labelled list for advisory model scoring."""
    conn = db.get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT pg_advisory_xact_lock(%s)", (724_913_210,))
            cur.execute(
                "SELECT * FROM simulation_runs WHERE id=%s AND status='draft' FOR UPDATE",
                (draft_id,),
            )
            if not cur.fetchone():
                raise ValueError("Reusable simulation list not found")
            if enabled:
                cur.execute(
                    "UPDATE simulation_runs SET is_validation_suite=FALSE WHERE status='draft'"
                )
            cur.execute(
                """UPDATE simulation_runs
                   SET is_validation_suite=%s,updated_at=now()
                   WHERE id=%s RETURNING *""",
                (bool(enabled), draft_id),
            )
            result = dict(cur.fetchone())
        conn.commit()
        return _public_run(result)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def evaluate_validation_suite(conn, version_id: str) -> dict:
    """Replay the designated independent labelled suite against one bundle."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT * FROM simulation_runs
               WHERE status='draft' AND is_validation_suite
               ORDER BY updated_at DESC LIMIT 1"""
        )
        row = cur.fetchone()
    if row is None:
        return {
            "configured": False,
            "passed": True,
            "reason": "No reusable event list is designated as the model validation suite.",
        }

    suite = _public_run(dict(row))
    factory = replay.load_bundle(version_id)
    evaluated_cases = []
    policy = replay.policy_snapshot()
    with runtime_config.override(policy):
        for stored_case in suite.get("cases") or []:
            try:
                result = replay._replay_range(conn, _replay_case(stored_case), factory)
            except Exception as exc:
                result = _case_error(stored_case, exc)
            evaluated_cases.append({**stored_case, "result": result})

    summary = replay.summarize(evaluated_cases)
    total = len(evaluated_cases)
    passed = (
        total > 0
        and summary.get("evaluated_ranges") == total
        and summary.get("correct_ranges") == total
        and summary.get("error_ranges") == 0
        and summary.get("unscored_ranges") == 0
        and summary.get("premature_critical_cases") == 0
        and summary.get("false_alert_cases") == 0
        and summary.get("timing_correct") == summary.get("timing_evaluated")
    )
    evidence = []
    for item in evaluated_cases:
        result = item.get("result") or {}
        timing = result.get("timing") or {}
        evidence.append({
            "description": item.get("description"),
            "machine_id": item.get("machine_id"),
            "start": item.get("start"),
            "end": item.get("end"),
            "expected_status": item.get("expected_status"),
            "predicted_status": result.get("predicted_status"),
            "match": result.get("match"),
            "expected_status_coverage": result.get("expected_status_coverage"),
            "timing_evaluable": timing.get("timing_evaluable"),
            "timing_pass": timing.get("timing_pass"),
            "error": result.get("error"),
        })
    return {
        "configured": True,
        "passed": passed,
        "suite_id": suite.get("id"),
        "suite_name": suite.get("name"),
        "model_version": version_id,
        "policy_snapshot": policy,
        "summary": summary,
        "cases": artifact_utils.to_json_safe(evidence),
    }


def delete_run(run_id: int) -> bool:
    conn = db.get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """DELETE FROM simulation_runs
                   WHERE id=%s AND status IN ('draft','completed','failed')""",
                (run_id,),
            )
            deleted = cur.rowcount == 1
        conn.commit()
        return deleted
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def recover_interrupted(conn=None) -> list[int]:
    owns_connection = conn is None
    conn = conn or db.get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM simulation_runs WHERE status IN ('queued','running') FOR UPDATE"
            )
            rows = cur.fetchall()
            queued = []
            for row in rows:
                run = _public_run(dict(row))
                queued.append(int(run["id"]))
                if run["status"] != "running":
                    continue
                cases = run.get("cases") or []
                for case in cases:
                    case["result"] = None
                    case["error"] = None
                    case["progress"] = None
                cur.execute(
                    """UPDATE simulation_runs
                       SET status='queued',cases=%s,summary=NULL,error=NULL,
                           completed_cases=0,started_at=NULL,finished_at=NULL,updated_at=now()
                       WHERE id=%s""",
                    (psycopg2.extras.Json(cases), run["id"]),
                )
        conn.commit()
        return queued
    except Exception:
        conn.rollback()
        raise
    finally:
        if owns_connection:
            conn.close()


def _case_error(case: dict, exc: Exception) -> dict:
    return {
        "expected_status": case["expected_status"],
        "predicted_status": "ERROR",
        "match": False,
        "evaluable": False,
        "error": str(exc),
        "explanation": str(exc),
    }


def _replay_case(case: dict) -> dict:
    item = dict(case)
    start = case.get("start") or case.get("event_start") or case.get("target")
    end = case.get("end") or case.get("event_end") or case.get("target")
    if not start or not end:
        raise ValueError("Simulation case has no labelled event range")
    item["event_start"] = replay.utc_datetime(start)
    item["event_end"] = replay.utc_datetime(end)
    return item


def _update_run(run_id: int, updater) -> dict | None:
    conn = db.get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM simulation_runs WHERE id=%s FOR UPDATE", (run_id,))
            row = cur.fetchone()
            if row is None:
                conn.rollback()
                return None
            run = _public_run(dict(row))
            updater(run)
            cur.execute(
                """UPDATE simulation_runs
                   SET status=%s,cases=%s,summary=%s,error=%s,completed_cases=%s,
                       started_at=%s,finished_at=%s,updated_at=now()
                   WHERE id=%s RETURNING *""",
                (
                    run["status"],
                    psycopg2.extras.Json(artifact_utils.to_json_safe(run.get("cases") or [])),
                    psycopg2.extras.Json(artifact_utils.to_json_safe(run["summary"])) if run.get("summary") is not None else None,
                    run.get("error"),
                    run.get("completed_cases", 0),
                    run.get("started_at"),
                    run.get("finished_at"),
                    run_id,
                ),
            )
            updated = dict(cur.fetchone())
        conn.commit()
        return _public_run(updated)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute(run_id: int) -> None:
    """Replay saved cases; the plant source remains read-only."""
    claimed = False

    def start(run: dict) -> None:
        nonlocal claimed
        if run.get("status") != "queued":
            return
        claimed = True
        run.update({
            "status": "running",
            "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "finished_at": None,
            "error": None,
            "summary": None,
            "completed_cases": 0,
        })
        for case in run.get("cases") or []:
            case["result"] = None
            case["error"] = None
            case["progress"] = None

    run = _update_run(run_id, start)
    if not claimed or not run or run.get("status") != "running":
        return

    conn = None
    try:
        conn = db.get_connection()
        runtime_config.load_from_db(conn)
        policy = run.get("policy_snapshot") or {}
        factory = replay.load_bundle(str(run["model_version"]))
        with runtime_config.override(policy):
            total_cases = max(1, len(run.get("cases") or []))
            for completed, stored_case in enumerate(run.get("cases") or [], start=1):
                last_progress_write = 0.0

                def publish_progress(value: dict) -> None:
                    nonlocal last_progress_write
                    now = time.monotonic()
                    phase = str(value.get("phase") or "source_replay")
                    if last_progress_write and now - last_progress_write < 1.5 and phase not in {"motion_profile", "finalizing"}:
                        return
                    last_progress_write = now
                    event_progress = max(0.0, min(1.0, float(value.get("event_progress") or 0.0)))
                    payload = artifact_utils.to_json_safe({
                        **value,
                        "status": "running",
                        "event_position": completed,
                        "event_count": total_cases,
                        "machine_id": stored_case.get("machine_id"),
                        "description": stored_case.get("description"),
                        "overall_progress": ((completed - 1) + event_progress) / total_cases,
                        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    })

                    def store_progress(current: dict) -> None:
                        current["cases"][completed - 1]["progress"] = payload

                    try:
                        _update_run(run_id, store_progress)
                    except Exception:
                        # Progress telemetry must not invalidate an otherwise
                        # correct read-only model evaluation.
                        pass

                try:
                    result = replay._replay_range(
                        conn,
                        _replay_case(stored_case),
                        factory,
                        progress_callback=publish_progress,
                    )
                except Exception as exc:
                    result = _case_error(stored_case, exc)
                safe_result = artifact_utils.to_json_safe(result)

                def save_case(current: dict, *, position=completed, value=safe_result) -> None:
                    case = current["cases"][position - 1]
                    case["result"] = value
                    case["error"] = value.get("error")
                    previous_progress = case.get("progress") or {}
                    case["progress"] = {
                        **previous_progress,
                        "status": "completed",
                        "phase": "completed",
                        "message": "Event replay completed.",
                        "event_progress": 1.0,
                        "overall_progress": position / total_cases,
                        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    }
                    current["completed_cases"] = position

                _update_run(run_id, save_case)

        current = get_run(run_id)
        if current is None:
            return
        report = replay.summarize(current.get("cases") or [])

        def finish(item: dict) -> None:
            item.update({
                "status": "completed",
                "summary": artifact_utils.to_json_safe(report),
                "completed_cases": item.get("total_cases", len(item.get("cases") or [])),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            })

        _update_run(run_id, finish)
    except Exception as exc:
        def fail(item: dict) -> None:
            item.update({
                "status": "failed",
                "error": str(exc),
                "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            })

        _update_run(run_id, fail)
    finally:
        if conn is not None:
            conn.close()
