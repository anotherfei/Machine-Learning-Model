"""Idempotent PostgreSQL migration for the compact ML control plane.

The application owns six tables in the dedicated ``ML`` schema. Historical
installations used thirteen public tables; ``migrate`` copies their data into
the consolidated layout and removes the superseded tables in one transaction.
"""
from __future__ import annotations

import json

import runtime_config


SCHEMA_NAME = "ML"
MIGRATION_LOCK_ID = 724_913_209

SCHEMA_SQL = r'''
CREATE SCHEMA IF NOT EXISTS "ML";

CREATE TABLE IF NOT EXISTS "ML".model_versions (
  version_id TEXT PRIMARY KEY,
  display_name TEXT,
  artifact_path TEXT NOT NULL,
  reference_signature TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('shadow','active','retired','rejected')),
  validation_report JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  promoted_at TIMESTAMPTZ,
  promoted_by TEXT
);
ALTER TABLE "ML".model_versions ADD COLUMN IF NOT EXISTS display_name TEXT;
ALTER TABLE "ML".model_versions DROP COLUMN IF EXISTS machine_calibrations;
CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_active
  ON "ML".model_versions ((status)) WHERE status='active';

CREATE TABLE IF NOT EXISTS "ML".spindle_predictions (
  id BIGSERIAL PRIMARY KEY,
  machine_id TEXT NOT NULL,
  tick_timestamp TIMESTAMPTZ NOT NULL,
  model_version TEXT NOT NULL,
  raw_reading JSONB NOT NULL,
  anomaly_score REAL NOT NULL,
  health_raw REAL NOT NULL,
  health_state REAL NOT NULL,
  trend_slope_per_day REAL NOT NULL,
  remaining_days REAL NOT NULL,
  failure_probability JSONB NOT NULL,
  maintenance_level TEXT NOT NULL,
  maintenance_reason TEXT NOT NULL,
  maintenance_trigger TEXT NOT NULL,
  top_contributors JSONB,
  feature_vector JSONB,
  is_backfill BOOLEAN NOT NULL DEFAULT FALSE,
  alert_status TEXT CHECK (alert_status IN ('pending','confirmed_anomaly','confirmed_normal')),
  alert_reviewed_by TEXT,
  alert_reviewed_at TIMESTAMPTZ,
  near_miss_status TEXT CHECK (near_miss_status IN ('pending','acknowledged','flagged')),
  near_miss_reviewed_by TEXT,
  near_miss_reviewed_at TIMESTAMPTZ,
  reference_candidate_at TIMESTAMPTZ,
  candidate_model_version TEXT REFERENCES "ML".model_versions(version_id) ON DELETE SET NULL,
  added_to_reference_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE(machine_id,tick_timestamp,model_version)
);
CREATE INDEX IF NOT EXISTS spindle_predictions_machine_ts_idx
  ON "ML".spindle_predictions(machine_id,tick_timestamp DESC);
CREATE INDEX IF NOT EXISTS spindle_predictions_alert_idx
  ON "ML".spindle_predictions(machine_id,alert_status,tick_timestamp DESC)
  WHERE alert_status IS NOT NULL;
CREATE INDEX IF NOT EXISTS spindle_predictions_candidate_idx
  ON "ML".spindle_predictions(candidate_model_version,reference_candidate_at)
  WHERE reference_candidate_at IS NOT NULL AND added_to_reference_at IS NULL;
CREATE INDEX IF NOT EXISTS spindle_predictions_near_miss_idx
  ON "ML".spindle_predictions(machine_id,near_miss_status,tick_timestamp DESC)
  WHERE near_miss_status IS NOT NULL;

CREATE TABLE IF NOT EXISTS "ML".retrain_jobs (
  id BIGSERIAL PRIMARY KEY,
  trigger TEXT NOT NULL CHECK (trigger IN ('manual','scheduled')),
  requested_by TEXT,
  force BOOLEAN NOT NULL DEFAULT FALSE,
  status TEXT NOT NULL CHECK (status IN ('queued','running','passed','rejected','failed','skipped')),
  candidate_signature TEXT,
  model_version_id TEXT REFERENCES "ML".model_versions(version_id) ON DELETE SET NULL,
  result JSONB,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS retrain_jobs_one_active
  ON "ML".retrain_jobs ((1)) WHERE status IN ('queued','running');
CREATE INDEX IF NOT EXISTS retrain_jobs_created_idx
  ON "ML".retrain_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS retrain_jobs_signature_idx
  ON "ML".retrain_jobs(candidate_signature,finished_at DESC);

CREATE TABLE IF NOT EXISTS "ML".simulation_runs (
  id BIGSERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  model_version TEXT REFERENCES "ML".model_versions(version_id) ON DELETE SET NULL,
  status TEXT NOT NULL CHECK (status IN ('draft','queued','running','completed','failed')),
  created_by TEXT NOT NULL,
  is_validation_suite BOOLEAN NOT NULL DEFAULT FALSE,
  cases JSONB NOT NULL DEFAULT '[]'::jsonb,
  policy_snapshot JSONB,
  total_cases INTEGER NOT NULL DEFAULT 0,
  completed_cases INTEGER NOT NULL DEFAULT 0,
  summary JSONB,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);
ALTER TABLE "ML".simulation_runs ADD COLUMN IF NOT EXISTS is_validation_suite BOOLEAN NOT NULL DEFAULT FALSE;
CREATE UNIQUE INDEX IF NOT EXISTS simulation_runs_one_active
  ON "ML".simulation_runs ((1)) WHERE status IN ('queued','running');
CREATE UNIQUE INDEX IF NOT EXISTS simulation_runs_one_validation_suite
  ON "ML".simulation_runs ((1)) WHERE status='draft' AND is_validation_suite;
CREATE INDEX IF NOT EXISTS simulation_runs_status_created_idx
  ON "ML".simulation_runs(status,created_at DESC);

CREATE TABLE IF NOT EXISTS "ML".app_users (
  username TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('viewer','admin')),
  disabled BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS "ML".state (
  namespace TEXT NOT NULL,
  key TEXT NOT NULL,
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by TEXT,
  PRIMARY KEY(namespace,key),
  CONSTRAINT state_namespace_check
    CHECK (namespace IN ('config','machine','operating_override','env_audit','migration','backfill'))
);
ALTER TABLE "ML".state DROP CONSTRAINT IF EXISTS state_namespace_check;
ALTER TABLE "ML".state ADD CONSTRAINT state_namespace_check
  CHECK (namespace IN ('config','machine','operating_override','env_audit','migration','backfill'));
CREATE INDEX IF NOT EXISTS state_namespace_updated_idx
  ON "ML".state(namespace,updated_at DESC);
'''


LEGACY_DROP_SQL = r'''
DROP TABLE IF EXISTS public.model_version_candidates CASCADE;
DROP TABLE IF EXISTS public.machine_model_calibrations CASCADE;
DROP TABLE IF EXISTS public.model_calibrations CASCADE;
DROP TABLE IF EXISTS public.reference_candidates CASCADE;
DROP TABLE IF EXISTS public.near_miss_reviews CASCADE;
DROP TABLE IF EXISTS public.alerts CASCADE;
DROP TABLE IF EXISTS public.machine_runtime_state CASCADE;
DROP TABLE IF EXISTS public.runtime_config CASCADE;
DROP TABLE IF EXISTS public.env_change_log CASCADE;
DROP TABLE IF EXISTS public.retrain_jobs CASCADE;
DROP TABLE IF EXISTS public.app_users CASCADE;
DROP TABLE IF EXISTS public.spindle_predictions CASCADE;
DROP TABLE IF EXISTS public.model_versions CASCADE;
DROP TABLE IF EXISTS public.regression_tests CASCADE;
'''


def _legacy_exists(cur, table: str) -> bool:
    cur.execute("SELECT to_regclass(%s)", (f"public.{table}",))
    return cur.fetchone()[0] is not None


def _reset_sequence(cur, qualified_table: str, column: str = "id") -> None:
    cur.execute(
        f'''SELECT setval(
              pg_get_serial_sequence('{qualified_table}','{column}'),
              COALESCE((SELECT max({column}) FROM {qualified_table}),1),
              EXISTS(SELECT 1 FROM {qualified_table})
            )'''
    )


def _migrate_legacy(cur) -> None:
    """Copy the former public control-plane tables into the five-table model."""
    if _legacy_exists(cur, "model_versions"):
        cur.execute(
            '''INSERT INTO "ML".model_versions
                 (version_id,artifact_path,reference_signature,status,validation_report,
                  created_at,promoted_at,promoted_by)
               SELECT version_id,artifact_path,reference_signature,status,validation_report,
                      created_at,promoted_at,promoted_by
               FROM public.model_versions
               ON CONFLICT(version_id) DO NOTHING'''
        )

    if _legacy_exists(cur, "spindle_predictions"):
        cur.execute(
            '''INSERT INTO "ML".spindle_predictions
                 (id,machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,
                  health_raw,health_state,trend_slope_per_day,remaining_days,
                  failure_probability,maintenance_level,maintenance_reason,
                  maintenance_trigger,top_contributors,is_backfill,created_at)
               SELECT id,machine_id,tick_timestamp,model_version,raw_reading,anomaly_score,
                      health_raw,health_state,trend_slope_per_day,remaining_days,
                      failure_probability,maintenance_level,maintenance_reason,
                      maintenance_trigger,top_contributors,is_backfill,created_at
               FROM public.spindle_predictions
               ON CONFLICT(machine_id,tick_timestamp,model_version) DO NOTHING'''
        )
        _reset_sequence(cur, '"ML".spindle_predictions')

    if _legacy_exists(cur, "alerts"):
        cur.execute(
            '''UPDATE "ML".spindle_predictions prediction
               SET alert_status=alert.status,
                   alert_reviewed_by=alert.reviewed_by,
                   alert_reviewed_at=alert.reviewed_at,
                   feature_vector=alert.feature_vector
               FROM (
                 SELECT DISTINCT ON (machine_id,tick_timestamp,model_version)
                        machine_id,tick_timestamp,model_version,status,reviewed_by,
                        reviewed_at,feature_vector,id
                 FROM public.alerts
                 ORDER BY machine_id,tick_timestamp,model_version,id DESC
               ) alert
               WHERE prediction.machine_id=alert.machine_id
                 AND prediction.tick_timestamp=alert.tick_timestamp
                 AND prediction.model_version=alert.model_version'''
        )

    if (
        _legacy_exists(cur, "reference_candidates")
        and _legacy_exists(cur, "alerts")
    ):
        has_staging = (
            _legacy_exists(cur, "model_version_candidates")
            and _legacy_exists(cur, "model_versions")
        )
        staged_expression = (
            "(SELECT mvc.version_id FROM public.model_version_candidates mvc "
            "JOIN public.model_versions mv ON mv.version_id=mvc.version_id "
            "WHERE mvc.candidate_id=rc.id AND mv.status='shadow' "
            "ORDER BY mv.created_at DESC LIMIT 1)"
            if has_staging
            else "NULL"
        )
        cur.execute(
            f'''UPDATE "ML".spindle_predictions prediction
                SET reference_candidate_at=rc.created_at,
                    added_to_reference_at=rc.added_to_reference_at,
                    candidate_model_version={staged_expression}
                FROM public.reference_candidates rc
                JOIN public.alerts alert ON alert.id=rc.alert_id
                WHERE prediction.machine_id=alert.machine_id
                  AND prediction.tick_timestamp=alert.tick_timestamp
                  AND prediction.model_version=alert.model_version'''
        )

    if _legacy_exists(cur, "near_miss_reviews"):
        cur.execute(
            '''UPDATE "ML".spindle_predictions prediction
               SET near_miss_status=review.status,
                   near_miss_reviewed_by=review.reviewed_by,
                   near_miss_reviewed_at=review.reviewed_at
               FROM public.near_miss_reviews review
               WHERE prediction.id=review.prediction_id'''
        )

    if _legacy_exists(cur, "retrain_jobs"):
        cur.execute(
            '''INSERT INTO "ML".retrain_jobs
                 (id,trigger,requested_by,force,status,candidate_signature,
                  model_version_id,result,error,created_at,started_at,finished_at)
               SELECT id,trigger,requested_by,force,status,candidate_signature,
                      model_version_id,result,error,created_at,started_at,finished_at
               FROM public.retrain_jobs
               ON CONFLICT(id) DO NOTHING'''
        )
        _reset_sequence(cur, '"ML".retrain_jobs')

    if _legacy_exists(cur, "app_users"):
        cur.execute(
            '''INSERT INTO "ML".app_users
                 (username,password_hash,role,disabled,created_at)
               SELECT username,password_hash,role,disabled,created_at
               FROM public.app_users
               ON CONFLICT(username) DO NOTHING'''
        )

    if _legacy_exists(cur, "runtime_config"):
        cur.execute(
            '''INSERT INTO "ML".state(namespace,key,value,updated_at,updated_by)
               SELECT 'config',key,value,updated_at,updated_by
               FROM public.runtime_config
               ON CONFLICT(namespace,key) DO UPDATE
                 SET value=EXCLUDED.value,updated_at=EXCLUDED.updated_at,
                     updated_by=EXCLUDED.updated_by'''
        )

    if _legacy_exists(cur, "machine_runtime_state"):
        cur.execute(
            '''INSERT INTO "ML".state(namespace,key,value,updated_at)
               SELECT 'machine',machine_id,
                      jsonb_build_object(
                        'operating_state',operating_state,
                        'reason',reason,
                        'confidence',confidence,
                        'activity_score',activity_score,
                        'stop_threshold',stop_threshold,
                        'run_threshold',run_threshold,
                        'tick_timestamp',tick_timestamp,
                        'state_changed_at',state_changed_at
                      ),updated_at
               FROM public.machine_runtime_state
               ON CONFLICT(namespace,key) DO UPDATE
                 SET value=EXCLUDED.value,updated_at=EXCLUDED.updated_at'''
        )

    if _legacy_exists(cur, "env_change_log"):
        cur.execute(
            '''INSERT INTO "ML".state(namespace,key,value,updated_at,updated_by)
               SELECT 'env_audit',id::text,
                      jsonb_build_object('changed_keys',changed_keys),
                      changed_at,changed_by
               FROM public.env_change_log
               ON CONFLICT(namespace,key) DO NOTHING'''
        )


def _assert_legacy_migrated(cur) -> None:
    """Abort before dropping a legacy table if any owned record was not copied."""
    direct_checks = (
        ("model_versions", '"ML".model_versions', "TRUE"),
        ("spindle_predictions", '"ML".spindle_predictions', "TRUE"),
        ("retrain_jobs", '"ML".retrain_jobs', "TRUE"),
        ("app_users", '"ML".app_users', "TRUE"),
        ("runtime_config", '"ML".state', "namespace='config'"),
        ("machine_runtime_state", '"ML".state', "namespace='machine'"),
        ("env_change_log", '"ML".state', "namespace='env_audit'"),
    )
    for legacy, target, condition in direct_checks:
        if not _legacy_exists(cur, legacy):
            continue
        cur.execute(f"SELECT count(*) FROM public.{legacy}")
        legacy_count = int(cur.fetchone()[0])
        cur.execute(f"SELECT count(*) FROM {target} WHERE {condition}")
        target_count = int(cur.fetchone()[0])
        if target_count < legacy_count:
            raise RuntimeError(
                f"ML schema migration refused to drop public.{legacy}: "
                f"copied {target_count} of {legacy_count} rows"
            )

    if _legacy_exists(cur, "alerts"):
        cur.execute(
            '''SELECT count(*) FROM public.alerts alert
               WHERE NOT EXISTS (
                 SELECT 1 FROM "ML".spindle_predictions prediction
                 WHERE prediction.machine_id=alert.machine_id
                   AND prediction.tick_timestamp=alert.tick_timestamp
                   AND prediction.model_version=alert.model_version
                   AND prediction.alert_status IS NOT NULL
               )'''
        )
        if int(cur.fetchone()[0]):
            raise RuntimeError(
                "ML schema migration found alert rows without matching predictions; "
                "legacy tables were left untouched"
            )

    if _legacy_exists(cur, "reference_candidates") and _legacy_exists(cur, "alerts"):
        cur.execute(
            '''SELECT count(*)
               FROM public.reference_candidates candidate
               JOIN public.alerts alert ON alert.id=candidate.alert_id
               WHERE NOT EXISTS (
                 SELECT 1 FROM "ML".spindle_predictions prediction
                 WHERE prediction.machine_id=alert.machine_id
                   AND prediction.tick_timestamp=alert.tick_timestamp
                   AND prediction.model_version=alert.model_version
                   AND prediction.reference_candidate_at IS NOT NULL
               )'''
        )
        if int(cur.fetchone()[0]):
            raise RuntimeError(
                "ML schema migration found retraining candidates without matching "
                "predictions; legacy tables were left untouched"
            )

    if _legacy_exists(cur, "near_miss_reviews"):
        cur.execute(
            '''SELECT count(*) FROM public.near_miss_reviews review
               WHERE NOT EXISTS (
                 SELECT 1 FROM "ML".spindle_predictions prediction
                 WHERE prediction.id=review.prediction_id
                   AND prediction.near_miss_status IS NOT NULL
               )'''
        )
        if int(cur.fetchone()[0]):
            raise RuntimeError(
                "ML schema migration found near-miss reviews without matching predictions; "
                "legacy tables were left untouched"
            )


def _apply_migration(conn) -> None:
    with conn.cursor() as cur:
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (MIGRATION_LOCK_ID,))
        cur.execute(SCHEMA_SQL)
        cur.execute(
            '''SELECT 1 FROM "ML".state
               WHERE namespace='migration' AND key='compact_schema_v1' '''
        )
        if cur.fetchone() is None:
            _migrate_legacy(cur)
            _assert_legacy_migrated(cur)
            cur.execute(LEGACY_DROP_SQL)
            cur.execute(
                '''INSERT INTO "ML".state(namespace,key,value)
                   VALUES('migration','compact_schema_v1','{"completed":true}'::jsonb)'''
            )

        cur.execute(
            '''SELECT 1 FROM "ML".state
               WHERE namespace='config' AND key='MAINTENANCE_URGENT_HORIZON_DAYS' '''
        )
        introducing_split_horizons = cur.fetchone() is None
        for key, value in runtime_config.defaults().items():
            cur.execute(
                '''INSERT INTO "ML".state(namespace,key,value)
                   VALUES('config',%s,%s::jsonb)
                   ON CONFLICT(namespace,key) DO NOTHING''',
                (key, json.dumps(value)),
            )
        cur.execute(
            '''UPDATE "ML".state SET value='1'::jsonb,updated_at=now()
               WHERE namespace='config' AND key='WORKER_POLL_SECONDS'
                 AND value='60'::jsonb AND updated_by IS NULL'''
        )
        if introducing_split_horizons:
            cur.execute(
                '''UPDATE "ML".state SET value='7'::jsonb,updated_at=now()
                   WHERE namespace='config' AND key='MAINTENANCE_HORIZON_DAYS'
                     AND value='1'::jsonb'''
            )
        cur.execute(
            '''UPDATE "ML".state SET value='7'::jsonb,updated_at=now()
               WHERE namespace='config' AND key='MAINTENANCE_HORIZON_DAYS'
                 AND value='30'::jsonb'''
        )


def migrate(conn) -> None:
    try:
        _apply_migration(conn)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    runtime_config.load_from_db(conn)
