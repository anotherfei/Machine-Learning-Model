# Architecture notes

## Initial model training

`train_isolation_forest.py` fits the Isolation Forest on a reference/
commissioning window. Where that window comes from is `config.REFERENCE_SOURCE`:

- **`"live"` (default)** — pulled directly from the same Postgres table
  and connection `worker.py`/`db.py` use in production, restricted to
  `[REFERENCE_WINDOW_START, REFERENCE_WINDOW_END)`. Training and
  production share one schema and one source of truth: no separate
  offline file whose columns can silently drift from what the live
  sensor actually emits. Those two timestamps must be set explicitly
  (`config.py`, or `--start`/`--end` on the command line) — the script
  refuses to guess a window from "whatever the table currently holds",
  since the table keeps growing and that would make every run pull a
  different, unreproducible reference set. Pin them once you've
  identified a confirmed-healthy commissioning stretch in the live data.
  Use `--all-machines` to discover all source machine IDs, or repeat
  `--machine-id` to select an explicit subset. Large ranges use two bounded
  streaming passes per machine. The first considers every clean source row
  while learning a bounded motion profile. Rare low-motion initializations and
  robust within-regime spread allow short genuine shutdown periods to remain
  detectable. Long commissioning ranges use a separate robust separation-
  quality floor because normal RUNNING speed/load variation is much broader
  over months than in the short live detector history; minimum center-ratio
  and chronological STOPPED confirmation rules remain unchanged.
  The second preserves the previous
  `WINDOW_SIZE-1` clean rows across database chunks, so rolling features are
  identical to full-trajectory feature construction, then automatically gates
  invalid/non-running/spec-ineligible rows. Every eligible row receives a
  deterministic random priority; the bounded per-machine reservoir therefore
  remains representative without loading the source range into memory. The
  retained pools are balanced to the same count before they are combined,
  preventing the highest-volume machine from dominating the shared forest.
  Initial scans run sequentially by default; `--machine-workers N` can execute
  independent machine tasks in spawned processes. Results are reassembled in
  source-machine order before normalization/fitting, and a failed or cancelled
  task terminates the complete process pool so no DB scan is orphaned.
  The newest balanced 20% (at least 20 rows per machine) is reserved as a
  forward validation holdout before normalizer, tree, or condition-anchor
  fitting. The completed model is tested independently on that holdout for
  finite score coverage and per-machine false-alert rate. If a labelled
  simulation suite is configured, that suite is replayed as a separate
  advisory score.
  `artifact_utils.save_artifacts()` records the fit and validation feature
  tables separately, plus machine counts and row identities, directly in a
  new immutable `artifacts/versions/<version_id>` bundle. The completed bundle
  is registered and selected through `active.json`; validation remains
  advisory and the artifact root is not a temporary duplicate.
- **`"csv"`** — the original offline-file behavior, reading
  `config.RAW_DATA_PATH`. Kept for offline experimentation and for CI/test
  fixtures that shouldn't need a reachable database.

For live training, the automatic machine-local operating-state pass excludes
`STOPPED` and `STARTING` rows without removing them before rolling feature
construction. The configured `SPEC_MAX` bounds (or `--full` to skip that test)
then decide which confirmed-running rows are accepted as healthy. The user
does not manually filter individual readings. Only use `--full` when every
running portion of the window is independently confirmed healthy.
`--include-non-running` is an explicit manual override for a window
independently known to contain running data only.

Before fitting the shared tree, every engineered feature is normalized with
that machine's confirmed-healthy median and robust IQR/MAD scale. The fitted
normalizers are part of the model bundle and are mandatory at inference; a
new machine cannot be scored until commissioning training includes it. Raw
reference features remain stored in physical units so a shadow retrain can
fit its own proposed normalizers without compounding an older transform.

After the shared tree is fitted, the trainer creates a robust residual
condition-score anchor for each machine from its normalized fit-row score
distribution. These anchors are automatic and switch with the model. Manual
web/CLI recalibration is deliberately unsupported because selecting data from
the current model's own `OK` decisions creates a circular, drifting baseline.

## Shared-model retraining

The active bundle carries `reference_features.csv`: the exact balanced,
machine-aware feature corpus used to fit its tree, and
`validation_features.csv`: equal per-machine confirmed-normal evidence that
was excluded from fitting and calibration. Shadow retraining starts from these
artifacts instead of trying to recover pooled features from a timestamp-only
CSV lookup. Legacy bundles create this separation on their first upgraded
retrain; newly commissioned bundles have it from initial training.

Only alerts that an operator marks `confirmed_normal` become retraining
candidates. The scheduler evaluates the batch and age thresholds per machine.
Its evaluation interval is runtime policy stored in PostgreSQL and editable by
an admin.
Candidates are cosine-deduplicated only against candidates from the same
machine, then merged into that machine's reference. Rebalancing gives every
machine the same row count and prioritizes new confirmed-normal rows when old
rows must be displaced. Retraining never onboards a new machine from alert
samples because those are a biased slice of its operating distribution. Add a
new machine by rerunning balanced initial training with a confirmed-healthy
commissioning window.

Each shadow model receives a confirmed-normal false-positive score for every
machine independently. When an Accuracy Simulation draft is designated as the
validation suite, the shadow also replays every labelled range and records
accuracy, coverage, lead-time checks, unscored/error ranges, false alerts, and
premature CRITICAL results. Both evaluations are advisory and do not block
promotion. Promotion switches the shared tree, machine normalizers, and
condition anchors together; the administrator uses the displayed evidence to
decide.

The scheduler and manual web action enqueue `retrain_jobs`; training does not
run inside the HTTP request. A PostgreSQL advisory lock plus the active-job
index enforce one fleet training process at a time. A completed shadow stages the
candidate prediction IDs through `spindle_predictions.candidate_model_version`.
They remain pending until an
administrator promotes that exact shadow, when they are consumed in the same
database transaction as activation. Deleting the shadow releases them. Only
one shadow can await a decision, so multiple proposals cannot reserve disjoint
candidate sets. Automatic retries suppress an unchanged validation rejection;
unexpected failures use the configured cooldown. Candidate IDs, active model,
material policy, pipeline hash, and retraining-protocol source hash form the
attempt fingerprint, so genuinely new evidence or deployed validation logic is
not blocked by an older result.

## Historical accuracy simulation

`"ML".simulation_runs` stores reusable model-independent event-list drafts, the
single optional model-validation-suite designation, queued work, progress, the
explicitly selected model version and runtime-policy snapshot, event evidence,
and the final report. One row owns the complete lifecycle so no separate case,
template, or result table is required. The scheduler runs only one replay at a
time. API restart requeues an interrupted run and clears its partial case
results before replaying it deterministically. A version with saved non-draft
simulation runs cannot be deleted, so the pinned report never loses its model
identity or immutable artifact bundle.

The replay reads the production PostgreSQL sensor source without modifying it;
only simulation lifecycle state is written to `"ML".simulation_runs`. Each case builds
the same per-machine motion gate and `SpindleMonitor` used by the worker, using
context before a seven-day causal lead window for motion profiling and warm-up.
The replay continues through the end of the labelled event. Pure rolling-feature,
normalization, and Isolation Forest phases use the shared bounded inference
batch size; stateful motion, condition, trend, and policy phases retain original
timestamp order. No source rows are sampled. Event accuracy
compares the human label with the time-weighted dominant production state
inside the range and requires at least 50% labelled-state duration coverage;
mean coverage is reported separately. Lead-time compliance uses event onset to check planning WARN timing,
urgent CRITICAL timing, premature escalation, and false alarms. This is
evaluation evidence only and cannot promote a model.

The active case publishes a throttled heartbeat into its existing JSON payload,
including phase, timeline progress, row count, throughput, elapsed time, ETA,
and the newest reached source timestamp. This makes slow model scoring visibly
different from a job that stopped reporting without adding another table.

## Production

`React/Vite -> FastAPI -> PostgreSQL` for control-plane requests and history.

The production control plane owns only six tables in the dedicated `"ML"`
schema: `spindle_predictions`, `model_versions`, `retrain_jobs`,
`simulation_runs`, `app_users`, and `state`. Prediction review/candidate lifecycle is one-to-one with its
originating prediction and therefore stays on that row. Machine calibrations
remain inside each immutable artifact bundle. The keyed `state` table
holds runtime policy, latest per-machine operating state, environment audit
records, bounded operator motion confirmations, and one replaceable sequential-catch-up progress document. The raw
sensor source remains in its existing schema.

The production worker runs separately and owns `SpindleMonitor`, feature engineering, anomaly scoring, health estimation, forecasting, and maintenance recommendation. Results are written to `"ML".spindle_predictions` and streamed to the UI.

An optional startup catch-up runs sequentially inside the production worker. It
captures a fixed source watermark, pins the active model, inserts only missing
`(machine, timestamp, model version)` predictions, and retains every bounded
per-machine runtime object created by the replay. Those exact rolling, Kalman,
trend, operating-state, and debounce objects continue into incremental polling,
so there is no second warm-up or state discontinuity at handoff. Larger bounded
source chunks, vectorized feature/model scoring, incremental trend statistics,
and batched inserts improve throughput without sampling readings or creating a
second inference implementation. Realtime calls the same batch-capable monitor
with a batch of one. An indexed bounded count supplies the historical total;
newer rows that arrive after the fixed watermark are reported and drained as a
separate open-ended phase.
The supervisor publishes `launching` before the worker starts. The job then
atomically publishes live progress to `artifacts/runtime/backfill_status.json`;
FastAPI reads that authoritative local document for the global frontend
notification so a busy PostgreSQL source cannot block observability. A legacy
`state` value is only a compatibility fallback when the document does not yet
exist. Worker connection attempts and preparation queries are bounded, so the
notification reaches `failed` with evidence rather than waiting indefinitely.
Model switches restart the replay. Mutable
inference/environment policy is locked by the API until handoff, and a compact
database/file revision fingerprint closes the startup race and restarts the
replay if that context changes.

## Temporary mock mode

`React/Vite -> FastAPI Demo/mock_main.py -> SQLite Demo/mock_demo.db`.

The production worker is not started. The mock API exposes the same frontend-facing endpoints and generates synthetic live ticks for multiple machine IDs so the web UI can be verified without a PostgreSQL server or trained artifacts.

## Local ports

- Frontend: `5173`
- FastAPI: `8000`
- PostgreSQL production default: `5432`

Database configuration is read from `.env`. Server command-line flags are not parsed as database overrides.
