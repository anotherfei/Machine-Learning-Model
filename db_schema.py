"""Idempotent database migration for the Spindle Condition Monitoring full-stack control plane."""
from __future__ import annotations
import json
import runtime_config

SCHEMA_SQL = r'''
CREATE TABLE IF NOT EXISTS alerts (
  id BIGSERIAL PRIMARY KEY,
  machine_id TEXT NOT NULL DEFAULT 'VVB001',
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
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'VVB001';
CREATE INDEX IF NOT EXISTS alerts_machine_status_idx ON alerts(machine_id, status, created_at DESC);

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

-- Health%-anchor recalibration runs (see recalibrate.py / recalibrate_service.py).
-- Each row is ONE computed baseline_mean/std anchor for a specific model
-- version, sourced from a recent live window; computing one does not
-- change what's deployed by itself. machine_model_calibrations (below) is
-- the per-machine on/off switch a human flips from the Models page.
CREATE TABLE IF NOT EXISTS model_calibrations (
  id BIGSERIAL PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
  machine_id TEXT NOT NULL DEFAULT 'VVB001',
  calibration JSONB NOT NULL,
  source_rows INTEGER NOT NULL,
  source_description TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by TEXT
);
ALTER TABLE model_calibrations ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'VVB001';
CREATE INDEX IF NOT EXISTS model_calibrations_version_idx ON model_calibrations(version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS model_calibrations_machine_version_idx ON model_calibrations(machine_id, version_id, created_at DESC);

-- NULL = use the pooled/default calibration baked into the bundle itself
-- ("normal"). Set = use that specific recalibration run's baseline
-- instead ("recalibrated"). See model_registry.promote()/set_calibration().
ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS active_calibration_id BIGINT REFERENCES model_calibrations(id);

CREATE TABLE IF NOT EXISTS machine_model_calibrations (
  machine_id TEXT NOT NULL,
  version_id TEXT NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
  calibration_id BIGINT NOT NULL REFERENCES model_calibrations(id) ON DELETE CASCADE,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(machine_id, version_id)
);
-- Preserve the old single-machine assignment when upgrading an existing DB.
INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id)
SELECT 'VVB001',version_id,active_calibration_id FROM model_versions
WHERE active_calibration_id IS NOT NULL
ON CONFLICT(machine_id,version_id) DO NOTHING;
UPDATE model_versions SET active_calibration_id=NULL WHERE active_calibration_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS regression_tests (
  id BIGSERIAL PRIMARY KEY,
  machine_id TEXT NOT NULL DEFAULT 'VVB001',
  description TEXT NOT NULL,
  timestamp_range TSTZRANGE NOT NULL,
  source_alert_id BIGINT REFERENCES alerts(id),
  minimum_anomaly_risk REAL NOT NULL DEFAULT 0.6,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE regression_tests ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'VVB001';

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
  machine_id TEXT NOT NULL DEFAULT 'VVB001',
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
ALTER TABLE spindle_predictions ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'VVB001';
ALTER TABLE spindle_predictions ADD COLUMN IF NOT EXISTS is_backfill BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS spindle_predictions_machine_ts_idx ON spindle_predictions(machine_id, tick_timestamp DESC);
DROP INDEX IF EXISTS spindle_predictions_tick_model_uq;
CREATE UNIQUE INDEX IF NOT EXISTS spindle_predictions_machine_tick_model_uq
  ON spindle_predictions(machine_id, tick_timestamp, model_version);

CREATE TABLE IF NOT EXISTS near_miss_reviews (
  id BIGSERIAL PRIMARY KEY,
  prediction_id BIGINT UNIQUE NOT NULL REFERENCES spindle_predictions(id) ON DELETE CASCADE,
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending','acknowledged','flagged')),
  reviewed_by TEXT,
  reviewed_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS near_miss_reviews_status_idx ON near_miss_reviews(status);
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
