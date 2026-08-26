"""Persistent, single-flight execution state for shadow retraining."""
from __future__ import annotations

import json

import psycopg2.errors
import psycopg2.extras

import db
import retrain_service
import runtime_config


ADVISORY_LOCK_ID = 724_913_207


def _row(conn, job_id: int) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM retrain_jobs WHERE id=%s", (job_id,))
        row = cur.fetchone()
    return dict(row) if row else None


def enqueue(conn, trigger: str, requested_by: str | None, force: bool) -> dict:
    if trigger not in ("manual", "scheduled"):
        raise ValueError("trigger must be manual or scheduled")
    pending = pending_shadow(conn)
    if pending:
        return {"queued": False, "reason": "shadow_awaiting_decision", "model": pending}
    signature = retrain_service.attempt_signature(conn)
    if not force:
        cooldown = float(runtime_config.get("RETRAIN_RETRY_COOLDOWN_HOURS"))
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM retrain_jobs
                   WHERE candidate_signature=%s AND status='rejected'
                   ORDER BY finished_at DESC LIMIT 1""",
                (signature,),
            )
            rejected = cur.fetchone()
            cur.execute(
                """SELECT * FROM retrain_jobs
                   WHERE candidate_signature=%s AND status='failed'
                     AND finished_at >= now()-(%s*interval '1 hour')
                   ORDER BY finished_at DESC LIMIT 1""",
                (signature, cooldown),
            )
            failed = cur.fetchone()
        blocked_by = rejected or failed
        if blocked_by:
            conn.rollback()
            return {
                "queued": False,
                "reason": "unchanged_rejected_evidence" if rejected else "retry_cooldown",
                "cooldown_hours": cooldown,
                "job": dict(blocked_by),
            }
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """INSERT INTO retrain_jobs(trigger,requested_by,force,status,candidate_signature)
                   VALUES(%s,%s,%s,'queued',%s) RETURNING *""",
                (trigger, requested_by, force, signature),
            )
            job = dict(cur.fetchone())
        conn.commit()
        return {"queued": True, "job": job}
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """SELECT * FROM retrain_jobs WHERE status IN ('queued','running')
                   ORDER BY created_at DESC LIMIT 1"""
            )
            active = cur.fetchone()
        return {"queued": False, "reason": "already_running", "job": dict(active) if active else None}


def execute(job_id: int) -> None:
    conn = db.get_connection()
    locked = False
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_ID,))
            locked = bool(cur.fetchone()[0])
        if not locked:
            with conn.cursor() as cur:
                cur.execute(
                    """UPDATE retrain_jobs SET status='skipped',finished_at=now(),
                              error='Another retraining process holds the fleet lock'
                       WHERE id=%s AND status='queued'""",
                    (job_id,),
                )
            conn.commit()
            return
        signature = retrain_service.attempt_signature(conn)
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE retrain_jobs SET status='running',started_at=now(),error=NULL,
                          candidate_signature=%s
                   WHERE id=%s AND status='queued' RETURNING force""",
                (signature, job_id),
            )
            row = cur.fetchone()
        conn.commit()
        if not row:
            return
        result = retrain_service.run_shadow_retrain(conn, force=bool(row[0]))
        if not result.get("started"):
            job_status = "skipped"
        elif result.get("status") == "shadow":
            job_status = "passed"
        else:
            job_status = "rejected"
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE retrain_jobs SET status=%s,model_version_id=%s,result=%s::jsonb,
                          finished_at=now() WHERE id=%s""",
                (job_status, result.get("version_id"), json.dumps(result, default=str), job_id),
            )
        conn.commit()
    except Exception as exc:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE retrain_jobs SET status='failed',error=%s,finished_at=now()
                   WHERE id=%s AND status IN ('queued','running')""",
                (str(exc), job_id),
            )
        conn.commit()
    finally:
        if locked:
            try:
                with conn.cursor() as cur:
                    cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_ID,))
                conn.commit()
            except Exception:
                conn.rollback()
        conn.close()


def recover_interrupted(conn) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_try_advisory_lock(%s)", (ADVISORY_LOCK_ID,))
        owns_lock = bool(cur.fetchone()[0])
        if owns_lock:
            cur.execute(
                """UPDATE retrain_jobs SET status='failed',finished_at=now(),
                          error='API process stopped while retraining'
                   WHERE status='running'"""
            )
            cur.execute("SELECT id FROM retrain_jobs WHERE status='queued' ORDER BY created_at")
            queued = [int(row[0]) for row in cur.fetchall()]
            cur.execute("SELECT pg_advisory_unlock(%s)", (ADVISORY_LOCK_ID,))
        else:
            # Another API/worker process is actively training. Do not mark its
            # job failed or schedule duplicate queued work from this process.
            queued = []
    conn.commit()
    return queued


def get(conn, job_id: int) -> dict | None:
    return _row(conn, job_id)


def list_recent(conn, limit: int = 20) -> list[dict]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM retrain_jobs ORDER BY created_at DESC LIMIT %s", (limit,))
        return [dict(row) for row in cur.fetchall()]


def queued_ids(conn) -> list[int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM retrain_jobs WHERE status='queued' ORDER BY created_at")
        return [int(row[0]) for row in cur.fetchall()]


def active_and_latest(conn) -> tuple[dict | None, dict | None]:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT * FROM retrain_jobs WHERE status IN ('queued','running')
               ORDER BY created_at DESC LIMIT 1"""
        )
        active = cur.fetchone()
        cur.execute(
            """SELECT * FROM retrain_jobs WHERE status NOT IN ('queued','running')
               ORDER BY finished_at DESC NULLS LAST,created_at DESC LIMIT 1"""
        )
        latest = cur.fetchone()
    return (dict(active) if active else None, dict(latest) if latest else None)


def pending_shadow(conn) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """SELECT version_id,created_at FROM model_versions
               WHERE status='shadow' ORDER BY created_at DESC LIMIT 1"""
        )
        row = cur.fetchone()
    return dict(row) if row else None
