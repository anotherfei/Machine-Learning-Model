"""Idempotent database migration for the Spindle Condition Monitoring full-stack control plane."""
from __future__ import annotations
import json
import runtime_config

SCHEMA_SQL = r'''
CREATE TABLE IF NOT EXISTS alerts (
  id BIGSERIAL PRIMARY KEY,
  tick_timestamp TIMESTAMPTZ NOT NULL,
  model_version TEXT NOT NULL,
  trigger TEXT NOT NULL CHECK (trigger IN ('health_threshold','health_inspect','trend_probability','none')),
  level TEXT NOT NULL CHECK (level IN ('WARN','CRITICAL')),
  health_state REAL NOT NULL,
  anomaly_score REAL NOT NULL,
  raw_reading JSONB NOT NULL,
  feature_vector JSONB,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','confirmed_anomaly','confirmed_normal')),
  reviewed_by TEXT,
  reviewed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS alerts_status_idx ON alerts(status, created_at DESC);

CREATE TABLE IF NOT EXISTS reference_candidates (
  id BIGSERIAL PRIMARY KEY,
  alert_id BIGINT UNIQUE REFERENCES alerts(id) ON DELETE CASCADE,
  tick_timestamp TIMESTAMPTZ NOT NULL,
  dedup_group_id BIGINT,
  added_to_reference_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS model_versions (
  version_id TEXT PRIMARY KEY,
  artifact_path TEXT NOT NULL,
  reference_signature TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('shadow','active','retired','rejected')),
  validation_report JSONB,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  promoted_at TIMESTAMPTZ,
  promoted_by TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS model_versions_one_active
  ON model_versions ((status)) WHERE status='active';

CREATE TABLE IF NOT EXISTS regression_tests (
  id BIGSERIAL PRIMARY KEY,
  description TEXT NOT NULL,
  timestamp_range TSTZRANGE NOT NULL,
  source_alert_id BIGINT REFERENCES alerts(id),
  minimum_anomaly_risk REAL NOT NULL DEFAULT 0.6,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS runtime_config (
  key TEXT PRIMARY KEY,
  value JSONB NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  updated_by TEXT
);

CREATE TABLE IF NOT EXISTS app_users (
  username TEXT PRIMARY KEY,
  password_hash TEXT NOT NULL,
  role TEXT NOT NULL CHECK (role IN ('viewer','admin')),
  disabled BOOLEAN NOT NULL DEFAULT FALSE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS env_change_log (
  id BIGSERIAL PRIMARY KEY,
  changed_by TEXT,
  changed_keys JSONB NOT NULL,
  changed_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS spindle_predictions (
  id BIGSERIAL PRIMARY KEY,
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
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE spindle_predictions ADD COLUMN IF NOT EXISTS is_backfill BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS spindle_predictions_ts_idx ON spindle_predictions(tick_timestamp DESC);
CREATE UNIQUE INDEX IF NOT EXISTS spindle_predictions_tick_model_uq ON spindle_predictions(tick_timestamp, model_version);
'''


def migrate(conn) -> None:
    with conn.cursor() as cur:
        cur.execute(SCHEMA_SQL)
        for key, value in runtime_config.defaults().items():
            cur.execute(
                "INSERT INTO runtime_config(key,value) VALUES(%s,%s::jsonb) ON CONFLICT(key) DO NOTHING",
                (key, json.dumps(value)),
            )
    conn.commit()
    runtime_config.load_from_db(conn)
