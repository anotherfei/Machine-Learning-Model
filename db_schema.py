"""Idempotent database migration for the Spindle Condition Monitoring full-stack control plane."""
from __future__ import annotations
import json
import runtime_config

SCHEMA_SQL = r'''
CREATE TABLE IF NOT EXISTS alerts (
  id BIGSERIAL PRIMARY KEY,
  machine_id TEXT NOT NULL DEFAULT 'MACHINE-001',
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
ALTER TABLE alerts ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'MACHINE-001';
ALTER TABLE alerts ALTER COLUMN machine_id SET DEFAULT 'MACHINE-001';
UPDATE alerts SET machine_id='MACHINE-001' WHERE machine_id='VVB001';
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

-- Confirmed-normal candidates remain pending while a validated shadow awaits
-- promotion. Deleting the shadow releases them; promotion consumes them.
CREATE TABLE IF NOT EXISTS model_version_candidates (
  version_id TEXT NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
  candidate_id BIGINT NOT NULL REFERENCES reference_candidates(id) ON DELETE CASCADE,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(version_id,candidate_id)
);
CREATE INDEX IF NOT EXISTS model_version_candidates_candidate_idx
  ON model_version_candidates(candidate_id);

-- Automatic robust condition-score anchors. Initial training and accepted
-- shadow retraining create exactly one anchor per commissioned machine and
-- model version; there is no operator-controlled recalibration path.
CREATE TABLE IF NOT EXISTS model_calibrations (
  id BIGSERIAL PRIMARY KEY,
  version_id TEXT NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
  machine_id TEXT NOT NULL DEFAULT 'MACHINE-001',
  calibration JSONB NOT NULL,
  source_rows INTEGER NOT NULL,
  source_description TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  created_by TEXT
);
ALTER TABLE model_calibrations ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'MACHINE-001';
ALTER TABLE model_calibrations ALTER COLUMN machine_id SET DEFAULT 'MACHINE-001';
UPDATE model_calibrations SET machine_id='MACHINE-001' WHERE machine_id='VVB001';
CREATE INDEX IF NOT EXISTS model_calibrations_version_idx ON model_calibrations(version_id, created_at DESC);
CREATE INDEX IF NOT EXISTS model_calibrations_machine_version_idx ON model_calibrations(machine_id, version_id, created_at DESC);

-- Legacy single-machine assignment retained only for idempotent migration.
ALTER TABLE model_versions ADD COLUMN IF NOT EXISTS active_calibration_id BIGINT REFERENCES model_calibrations(id);

CREATE TABLE IF NOT EXISTS machine_model_calibrations (
  machine_id TEXT NOT NULL,
  version_id TEXT NOT NULL REFERENCES model_versions(version_id) ON DELETE CASCADE,
  calibration_id BIGINT NOT NULL REFERENCES model_calibrations(id) ON DELETE CASCADE,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY(machine_id, version_id)
);
INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id,updated_at)
SELECT 'MACHINE-001',version_id,calibration_id,updated_at
FROM machine_model_calibrations WHERE machine_id='VVB001'
ON CONFLICT(machine_id,version_id) DO NOTHING;
DELETE FROM machine_model_calibrations WHERE machine_id='VVB001';
-- Preserve the old single-machine assignment when upgrading an existing DB.
INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id)
SELECT 'MACHINE-001',version_id,active_calibration_id FROM model_versions
WHERE active_calibration_id IS NOT NULL
ON CONFLICT(machine_id,version_id) DO NOTHING;
UPDATE model_versions SET active_calibration_id=NULL WHERE active_calibration_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS regression_tests (
  id BIGSERIAL PRIMARY KEY,
  machine_id TEXT NOT NULL DEFAULT 'MACHINE-001',
  description TEXT NOT NULL,
  timestamp_range TSTZRANGE NOT NULL,
  source_alert_id BIGINT REFERENCES alerts(id),
  target_prediction_id BIGINT,
  target_timestamp TIMESTAMPTZ,
  minimum_anomaly_risk REAL NOT NULL DEFAULT 0.6,
  created_by TEXT,
  disabled_at TIMESTAMPTZ,
  disabled_by TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
ALTER TABLE regression_tests ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'MACHINE-001';
ALTER TABLE regression_tests ALTER COLUMN machine_id SET DEFAULT 'MACHINE-001';
UPDATE regression_tests SET machine_id='MACHINE-001' WHERE machine_id='VVB001';
ALTER TABLE regression_tests ADD COLUMN IF NOT EXISTS disabled_at TIMESTAMPTZ;
ALTER TABLE regression_tests ADD COLUMN IF NOT EXISTS disabled_by TEXT;
ALTER TABLE regression_tests ADD COLUMN IF NOT EXISTS created_by TEXT;
ALTER TABLE regression_tests ADD COLUMN IF NOT EXISTS target_prediction_id BIGINT;
ALTER TABLE regression_tests ADD COLUMN IF NOT EXISTS target_timestamp TIMESTAMPTZ;

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
  machine_id TEXT NOT NULL DEFAULT 'MACHINE-001',
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

CREATE TABLE IF NOT EXISTS retrain_jobs (
  id BIGSERIAL PRIMARY KEY,
  trigger TEXT NOT NULL CHECK (trigger IN ('manual','scheduled')),
  requested_by TEXT,
  force BOOLEAN NOT NULL DEFAULT FALSE,
  status TEXT NOT NULL CHECK (status IN ('queued','running','passed','rejected','failed','skipped')),
  candidate_signature TEXT,
  model_version_id TEXT REFERENCES model_versions(version_id) ON DELETE SET NULL,
  result JSONB,
  error TEXT,
  created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  started_at TIMESTAMPTZ,
  finished_at TIMESTAMPTZ
);
CREATE UNIQUE INDEX IF NOT EXISTS retrain_jobs_one_active
  ON retrain_jobs ((1)) WHERE status IN ('queued','running');
CREATE INDEX IF NOT EXISTS retrain_jobs_created_idx ON retrain_jobs(created_at DESC);
CREATE INDEX IF NOT EXISTS retrain_jobs_signature_idx ON retrain_jobs(candidate_signature,finished_at DESC);
ALTER TABLE spindle_predictions ADD COLUMN IF NOT EXISTS machine_id TEXT NOT NULL DEFAULT 'MACHINE-001';
ALTER TABLE spindle_predictions ALTER COLUMN machine_id SET DEFAULT 'MACHINE-001';
DELETE FROM spindle_predictions legacy
USING spindle_predictions current
WHERE legacy.machine_id='VVB001' AND current.machine_id='MACHINE-001'
  AND legacy.tick_timestamp=current.tick_timestamp
  AND legacy.model_version=current.model_version
  AND legacy.id<>current.id;
UPDATE spindle_predictions SET machine_id='MACHINE-001' WHERE machine_id='VVB001';
ALTER TABLE spindle_predictions ADD COLUMN IF NOT EXISTS is_backfill BOOLEAN NOT NULL DEFAULT FALSE;
CREATE INDEX IF NOT EXISTS spindle_predictions_machine_ts_idx ON spindle_predictions(machine_id, tick_timestamp DESC);
DROP INDEX IF EXISTS spindle_predictions_tick_model_uq;
CREATE UNIQUE INDEX IF NOT EXISTS spindle_predictions_machine_tick_model_uq
  ON spindle_predictions(machine_id, tick_timestamp, model_version);

-- Repair regression floors created by older builds that treated sklearn's
-- raw (usually negative) score_samples value as a 0-1 anomaly probability.
-- Only the telltale clamped 0.05 auto-generated rows are changed; manual
-- regression policy is never rewritten.
UPDATE regression_tests rt
SET minimum_anomaly_risk=GREATEST(0.05,LEAST(0.95,1.0-p.health_state/100.0))
FROM spindle_predictions p
WHERE rt.minimum_anomaly_risk=0.05
  AND rt.description LIKE 'Near-miss%prediction #%'
  AND substring(rt.description from 'prediction #([0-9]+)$') IS NOT NULL
  AND p.id=(substring(rt.description from 'prediction #([0-9]+)$'))::BIGINT;

-- Pin auto-generated near-miss tests to the exact reviewed prediction. Older
-- builds only stored a broad timestamp range, which allowed an unrelated peak
-- elsewhere in that range to hide a false negative at the flagged tick.
UPDATE regression_tests rt
SET target_prediction_id=p.id,target_timestamp=p.tick_timestamp
FROM spindle_predictions p
WHERE rt.target_prediction_id IS NULL
  AND rt.description LIKE 'Near-miss%prediction #%'
  AND substring(rt.description from 'prediction #([0-9]+)$') IS NOT NULL
  AND p.id=(substring(rt.description from 'prediction #([0-9]+)$'))::BIGINT;
CREATE INDEX IF NOT EXISTS regression_tests_target_prediction_idx
  ON regression_tests(target_prediction_id) WHERE target_prediction_id IS NOT NULL;

-- Latest automatically inferred motion state for each machine. This is
-- deliberately separate from maintenance_level: STOPPED/STARTING describe
-- whether ML scoring is applicable, not the health of the spindle.
CREATE TABLE IF NOT EXISTS machine_runtime_state (
  machine_id TEXT PRIMARY KEY,
  operating_state TEXT NOT NULL CHECK (operating_state IN ('UNKNOWN','RUNNING','STOPPED','STARTING','SENSOR_FAULT')),
  reason TEXT NOT NULL,
  confidence REAL NOT NULL DEFAULT 0,
  activity_score REAL,
  stop_threshold REAL,
  run_threshold REAL,
  tick_timestamp TIMESTAMPTZ NOT NULL,
  state_changed_at TIMESTAMPTZ NOT NULL,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

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
