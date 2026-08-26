# Spindle Condition Monitoring

Unsupervised multi-machine spindle condition monitoring with a FastAPI control plane and React/TypeScript web interface.

Production inference remains label-free: `health_status` is not used by the training or realtime inference path.

## Quick web demo — no PostgreSQL or model artifacts required

First-time setup:

```powershell
.\setup_local.ps1
```

Start the temporary mock demo:

```powershell
.\start_project.ps1 -Mock
```

Open:

- Web UI: `http://localhost:5173`
- API docs: `http://localhost:8000/docs`

Default mock login:

```text
username: admin
password: change-me-on-first-deployment
```

Mock mode is isolated under `Demo/` and creates `Demo/mock_demo.db`. It contains seeded users, sensor/prediction history, pending alerts, model versions, thresholds, and mock environment settings. The mock API also emits a synthetic live tick once per second through the same `/ws/live` endpoint used by the frontend. Deleting `Demo/` removes the demo implementation and data without affecting normal production startup.

Reset the temporary database:

```powershell
.\Demo\reset_mock.ps1
```

Then start again with `-Mock`.

## Production mode

Configure `.env` with the real PostgreSQL connection, then train. The trainer writes a complete immutable bundle directly under `artifacts/versions/<version_id>`, records its advisory validation report, and registers the completed version. For a shared model, select every confirmed-healthy machine in the commissioning window:

```powershell
python train_isolation_forest.py --source live --all-machines --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full
```

Machine scans are sequential by default. On a multicore workstation, process
two machines concurrently with `--machine-workers 2`:

```powershell
python train_isolation_forest.py --source live --all-machines --machine-workers 2 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full
```

Start with two workers. Each worker owns separate PostgreSQL connections,
rolling-feature state, and bounded reservoirs; using four can increase database
load and memory substantially. Parallel results are restored to source-machine
order before balancing, so completion timing does not change the model input.

This pulls a pinned commissioning window from the same PostgreSQL table `worker.py` reads. No manual row query or value filter is required. The trainer scans every selected source row in bounded, cancellable chunks. Its first pass learns stationary and rotating vibration regimes independently for every machine; its second pass preserves rolling-window continuity and automatically excludes invalid, `STOPPED`, and `STARTING` rows. `--full` trusts all remaining confirmed-running rows as healthy; omit it only after genuine `SPEC_MAX` bounds have been configured. Every eligible feature row receives an equal deterministic opportunity to enter a bounded reservoir (100,000 rows per machine by default), so tens of millions of readings are considered without being loaded into memory together. Each machine then contributes the same retained count. The newest balanced 20% (at least 20 rows per machine) is saved as validation evidence and excluded from normalizer fitting, model fitting, and condition calibration. Each remaining engineered feature is converted to a robust machine-relative value `(feature - healthy median) / healthy IQR-scale`, and one Isolation Forest is fitted to the balanced relative-feature pool. The raw fit and validation features remain stored separately in physical units for audit and future retraining. The trainer also creates a robust automatic condition-score anchor for each machine from fit rows only.

Preview the exact normalization, operating-state counts, thresholds, and balanced row counts without fitting or writing artifacts:

```powershell
python train_isolation_forest.py --source live --all-machines --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full --preview-reference
```

If a selected window contains only running data and therefore has no separable stationary cluster, `--include-non-running` is the explicit manual override. Do not use that override on a mixed production window.

`--all-machines` discovers IDs through `PG_COL_MACHINE_ID`. To train on an explicit subset instead, repeat `--machine-id`:

```powershell
python train_isolation_forest.py --source live --machine-id MACHINE-001 --machine-id MACHINE-002 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full
```

The selected period's running portions must be confirmed healthy for every included machine. The script intentionally does not invent a date range. Set `REFERENCE_WINDOW_START`/`REFERENCE_WINDOW_END` in `config.py` or pass `--start`/`--end`. Database scanning is uncapped, but retained in-memory fit/validation evidence is safely bounded by `TRAINING_MAX_ROWS_PER_MACHINE`; `--max-rows-per-machine` is an optional one-run override and does not truncate the database scan. The automatic selection and complete scan counts are recorded in `metadata.json`. See `architecture-notes.md` → "Initial model training" for the offline-CSV fallback (`--source csv`).

### Feature diagnostics

After training, inspect the shared model's effective feature and sensor weighting with:

```powershell
python feature_diagnostics.py
```

The default diagnostic takes a deterministic, balanced sample of 5,000 reference rows per machine, performs five shuffles per feature, and evaluates those shuffles concurrently with up to four workers. Its live progress bar reports completed evaluations, throughput, and estimated remaining time. Sampling affects only this read-only report; it does not alter the trained model or its artifacts.

Tune the diagnostic workload explicitly when needed:

```powershell
python feature_diagnostics.py --rows-per-machine 10000 --permutations 10 --workers 4
```

An exhaustive run remains available with `--all-rows`, but it can be much slower and temporarily consumes one shuffled feature matrix per active worker. Use fewer workers if memory is constrained. Run `python feature_diagnostics.py --help` for all options.

### PostgreSQL application schema

Production control-plane data lives in the dedicated uppercase `"ML"` schema, separate from the existing raw sensor table. The application uses six tables: `spindle_predictions`, `model_versions`, `retrain_jobs`, `simulation_runs`, `app_users`, and `state`. Alert reviews, near-miss reviews, and retraining-candidate lifecycle fields are stored on their originating prediction; reusable simulation lists and their eventual results share `simulation_runs`; thresholds, machine runtime state, environment audit entries, and the single current backfill-progress document share the keyed `state` table. Machine calibrations and normalizers live only inside each immutable version folder, avoiding a second database copy that could drift from the bundle selected by `active.json`.

On the first startup after upgrading, the idempotent migration creates `"ML"`, copies records from the former public control-plane tables, verifies that owned records were preserved, and removes those superseded tables in the same transaction. A failed verification rolls the transaction back and leaves the legacy tables untouched. The configured production sensor table is never moved or modified by this migration.

### PostgreSQL index required for long training windows

Live training filters each machine by `mach_id`, restricts the selected `datetime` range, and reads it in chronological order. Large source tables therefore need a composite index with the machine column first and timestamp second. For the default production mapping, run [`database/create_training_source_index.sql`](database/create_training_source_index.sql) once using DBeaver, `psql`, or the database administration tool. Its `CREATE INDEX CONCURRENTLY` statement is designed not to block normal sensor ingestion, but it must be run outside an explicit transaction and should still be scheduled with the database administrator because a large index build consumes storage and server resources. If your `.env` maps different table or column names, change the SQL identifiers to match.

Worker startup uses the same composite index to fetch only the newest motion-profile and warm-up rows for each discovered machine. It does not rank or scan the complete source table. If startup stops before printing `Operating-state calibration`, verify that the composite source index exists and that PostgreSQL's plan uses it.

Machine discovery uses a recursive loose index scan over the leading machine-ID column. This avoids a full `SELECT DISTINCT` pass over every sensor row and keeps `/api/machines`, fleet startup, and worker initialization responsive as the source table grows.

Once caught up, the production worker checks for new source rows every second by default. When rows are available it drains successive batches without sleeping, so a restart backlog does not add one polling delay per batch. `WORKER_POLL_SECONDS` remains administrator-editable under Global thresholds → Machine availability.

The final read-only `EXPLAIN` in that file should show `p1_sel5_vibration_mach_id_datetime_idx` (or another equivalent composite index), rather than a sequential scan. Training now reports progress every roughly 500,000 source rows and limits each read transaction to a daily slice. A two-minute per-fetch statement timeout prevents an unindexed query from hanging indefinitely. Neither Isolation Forest CPU parallelism nor a GPU can accelerate database I/O; model fitting itself uses all CPU cores. It is safe to cancel because model artifacts are written only after scanning, feature engineering, balancing, and fitting have completed.

Then run:

```powershell
.\start_project.ps1
```

To replay the newest 30 days in the configured sensor source and then continue
realtime monitoring from that exact warmed state, use:

```powershell
.\start_project.ps1 -BackfillDays 30
```

`-Days 30` is an equivalent short alias. The range ends at the newest source
timestamp, so archived sources are handled correctly. This is an explicit
historical replay through the same model, operating-state, condition, forecast,
and maintenance pipeline used live. One worker processes the bounded history,
retains its machine-specific rolling features, operating-state detectors,
Kalman filters, trends, and maintenance debounce state, and then drains rows
that arrived after the captured watermark before settling into normal polling.
A global website notification shows percentage, current machine, rows,
total historical rows, throughput, and ETA on every page. The fixed historical
range is counted through the source index; if that read times out, the UI marks
its continuously refined total as an estimate instead of blocking catch-up.
Rows newer than the captured end are then shown separately during the open-ended
drain phase. Its preparation state is published before
machine discovery and source-watermark queries begin, so slow PostgreSQL setup
work is visible immediately. Progress is published atomically through
`artifacts/runtime/backfill_status.json`, so the notification endpoint does not
compete with historical PostgreSQL reads. This local document is the live status
authority; the former `state` entry is read only as a compatibility fallback
when no document exists. The supervisor publishes `launching` before the worker
starts, followed by explicit `connecting` and per-machine `preparing` stages.
Preparation queries use the configured 120-second database statement timeout,
so a missing/ineffective source index becomes a visible failure instead of an
unbounded wait. The terminal prints only lifecycle messages.
Rows already predicted by the same model version remain unchanged in
PostgreSQL, although their raw inputs are still evaluated to reconstruct the
exact state needed for later rows. Rolling features and Isolation Forest scores
are vectorized per bounded chunk, large forest batches use CPU tree parallelism,
and condition/trend/maintenance state is still applied chronologically per row.
Existing same-model timestamps bypass redundant JSON encoding and conflict
inserts. These optimizations do not sample data or change the active model. Model promotion
restarts catch-up against the new bundle, while inference-policy and environment
edits are locked by the API until handoff. A continuity fingerprint also catches
changes made during the narrow startup boundary and restarts safely, so state
cannot mix configurations. Raw sensor
snapshots and the website remain available during catch-up, but fresh realtime
ML status waits until the worker reaches the watermark. It can still take a
long time on a high-frequency source because every reading is evaluated. The
warmed state is intentionally process-local: if the worker or computer stops,
the next launch replays the selected interval again (already stored same-model
predictions are preserved). A raw row inserted later with a timestamp behind
the completed watermark also requires another catch-up run to affect state.

Production mode starts:

1. FastAPI on `127.0.0.1:8000`
2. the supervised worker process, optionally beginning with sequential catch-up
3. React/Vite on `localhost:5173`

The worker owns the actual `SpindleMonitor` inference path. The frontend does not duplicate ML or maintenance logic. The supervised API completes schema migration before the launcher starts the worker, so that worker skips a redundant second migration; standalone `python worker.py` still migrates defensively. `local_launcher.py` supervises it directly and restarts an interrupted catch-up when an active-model switch requires a clean replay; no second supervisor wrapper is required.

The production dashboard discovers machine IDs and polls the latest sensor row directly from the configured PostgreSQL source. This keeps real sensor channels visible while the ML worker warms up or reconnects. Prediction WebSocket messages add model health and maintenance results when available. The worker still processes and stores every source prediction, while the machine dashboard renders only the newest result on a five-second cadence. Review alerts are event records: they are created only when maintenance severity escalates into WARN or CRITICAL, not once per prediction tick. Source discovery failures are shown explicitly in the UI instead of silently displaying a fake/default machine.

The web console separates scope in its navigation. **Home**, Models & retraining, Global thresholds, and Environment are fleet-wide. Home includes a seven-day fleet recap built from hourly average condition scores, with a separate color-coded line and legend entry for every current source machine; machines are never averaged together. Selecting a machine opens that machine's own Overview, Status review, and History workspace. The machine selector no longer appears as a global top-bar filter, and every settings screen states whether a change applies fleet-wide or to one machine. Global thresholds exposes only runtime-writable policy: condition-score sensitivity and warm-up, condition/probability boundaries, real-time trend lookback and confidence, trend stabilization/confirmation/recovery, source polling, shutdown/restart confirmation, source-staleness timeout, and near-miss review analysis. Every control has an info icon describing its purpose and the effect of raising or lowering it; model-bound feature and normalization settings are intentionally absent. Settings that change the condition scale or buffer sizes reset in-memory monitors at a tick boundary so old and new policy state is never blended.

Normal worker startup reads only a recent warm-up tail for each machine and continues from the current source watermark. Supplying `-BackfillDays` deliberately selects the slower sequential catch-up path described above.

### Multiple machines

The sensor source may contain multiple machines in one table. Set these values in `.env`:

```text
PG_COL_MACHINE_ID=machine_id
DEFAULT_MACHINE_ID=MACHINE-001
```

`VVB001` is the shared ifm sensor model and continues to define the sensor channels and valid ranges. It is not used as a machine ID. `PG_COL_MACHINE_ID` must point to the separate source-table column containing asset identities such as `MACHINE-001`, `LINE-A-SPINDLE-02`, or your plant's own naming scheme; the website discovers its selector options from those values.

Source machine IDs are displayed without surrounding whitespace, while database filtering keeps direct equality predicates so PostgreSQL can use normal machine/timestamp indexes on large source tables. When switching between Demo and production, the frontend clears the previous session's machine list and waits for `/api/machines` before starting live polling. One non-overlapping fleet-snapshot request runs once per minute and supplies Home, the machine navigation, Overview fallback data, and Status Review; its backend performs machine lookups sequentially on one connection instead of issuing a concurrent query per machine. Overview adds a WebSocket for live detail without starting another HTTP polling loop.

The VVB001 IO-Link process value for `v-RMS` arrives from PostgreSQL in m/s with 0.0001 m/s resolution. `db.canonical_sensor_reading()` converts it once to the application's canonical mm/s unit (`v_rms_mms`, scale ×1000). The raw PostgreSQL table is never modified. Training, realtime inference, backfill, retraining, the API, and the website all consume the same canonical value.

Every `(timestamp, machine_id)` row is polled in order. The worker creates an independent rolling feature window, Kalman filter, trend history, and maintenance debouncer for each machine, while all machines use the active model bundle. Trend slopes, remaining-life forecasts, failure probabilities, alert context, and near-miss windows use actual database timestamps; they do not assume one source row equals one minute. Predictions, alerts, near-miss calculations, history, and live WebSocket messages are isolated by `machine_id`.

Retraining policy also exposes the automatic eligibility-check interval and the timestamp range captured around a flagged near miss. Environment masks PostgreSQL passwords, the application signing secret, and the bootstrap administrator password. `APP_SESSION_SECONDS` controls newly issued login lifetimes and is validated between 5 minutes and 7 days. Production startup automatically replaces an absent or public placeholder signing key with a cryptographically generated persistent key; weak signing keys, bootstrap passwords, and newly created user passwords are rejected. Both REST requests and the live WebSocket require a valid signed session.

### Automatic stopped/running detection without a PLC

The production worker also creates an independent operating-state detector for each machine. It learns that machine's low- and high-vibration regimes from recent acceleration RMS, velocity RMS, and acceleration peak history. It only enables automatic STOPPED detection when those regimes are clearly separated; otherwise it reports `UNKNOWN` and leaves ML monitoring enabled rather than guessing.

The runtime states are:

- `UNKNOWN`: not enough evidence for a safe motion threshold; ML remains enabled.
- `RUNNING`: vibration is in the learned rotating regime.
- `STOPPED`: sustained low vibration is in the learned stationary regime. Raw sensors continue to be polled, but ML scoring, alerts, and retraining inputs are suppressed.
- `STARTING`: rotation returned and the rolling feature/Kalman/trend state is warming from a clean reset.
- `SENSOR_FAULT`: a required channel is missing, invalid, non-finite, or a motion magnitude is negative. Constant readings are not classified as a fault because a genuinely stopped dedicated sensor can be flat.
- `NO_DATA`: the API has not received a fresh source row within `SOURCE_STALE_SECONDS`; this state is derived when the dashboard reads the source.

This detects *stationary versus rotating*, not electrical power. Without a PLC, drive-current contact, or other independent run signal, software cannot prove that a stationary spindle is electrically OFF. The worker keeps reading while `STOPPED` so it can detect an automatic restart. On confirmed shutdown it discards that machine's rolling features, Kalman state, trend history, and maintenance debouncer; after restart it emits no ML result until the fresh pipeline is warm.

An administrator can also use **Confirm manually** on a machine Overview when an operator has physically checked the spindle. The confirmation can set live motion to `RUNNING` or `STOPPED` for 15 minutes through 24 hours, records who made it and an optional note, expires automatically, and can be cleared early. The worker retains and displays the automatic estimate alongside the effective operator-confirmed state. A manual confirmation affects only live inference and never rewrites history or simulations. `NO_DATA` and `SENSOR_FAULT` remain higher-priority safety gates and cannot be hidden by an operator confirmation. Confirmations reuse the consolidated `"ML".state` table; no additional table is created.

The displayed percentage is a relative **condition score**, not measured physical remaining health. Automatic machine anchors use the healthy score median and a robust MAD/IQR spread, with a small minimum spread so nearly constant reference scores cannot amplify numerical noise. Manual web recalibration was intentionally removed: a current model selecting its own `OK` rows is circular and can silently shift maintenance decisions.

Shadow retraining follows the same shared-model rule. Confirmed-normal alerts are grouped and deduplicated in each machine's relative feature space, new rows replace older rows without allowing one machine to dominate, and fresh robust normalizers are fitted on the balanced training side only. On the first upgraded retrain, a fixed 20% machine-balanced holdout (at least 20 rows per machine) is reserved from previously accepted baseline rows and excluded from the shadow's normalization, fitting, and calibration; every new reviewed candidate stays on the training side. The bootstrap active model may have seen those legacy rows, so that first comparison is deliberately conservative. After the first promotion, the holdout is preserved outside the active and future shadow fit sets. False-positive validation is always evaluated per machine so a good fleet average cannot hide a regression on one machine. A promotable shadow bundles its fit reference, held-out validation evidence, fresh normalizers, and automatic condition anchors so the complete context switches atomically with the model.

Manual historical re-prediction keeps predictions and Status Review alerts consistent in bounded transactions. Pending machine-generated alerts are updated, created, or removed to match replayed severity transitions; repeated predictions at the same WARN or CRITICAL level do not become separate alerts. Predictions tied to an already reviewed alert or near-miss are not rewritten, preserving the exact evidence behind the human decision. Startup sequential catch-up is stricter: an existing `(machine, timestamp, model version)` result is never updated, while its input still contributes to the in-memory state handed directly to realtime. The live API also withholds condition and maintenance values whenever the newest PostgreSQL sensor row is newer than the worker's prediction, and the frontend shows that the worker is catching up instead of combining mismatched timestamps.

Forecast maintenance risk uses the Brownian first-passage probability of crossing the critical condition boundary at any time within the selected horizon. Its diffusion scale comes from detrended condition innovations and actual timestamp gaps, not from the regression residual level. The seven-day horizon is planning evidence and can produce only `WARN`; trend-based `CRITICAL` requires the separately configured urgent horizon, which is constrained to one day or less and must remain shorter than the planning horizon. Directly critical condition evidence can still become `CRITICAL`. Both escalation paths pass through timestamp-based confirmation (`WARN` first, then `CRITICAL`) to prevent a single noisy reading from changing maintenance state. These values remain model-estimated risk; validate their numeric calibration against real maintenance and failure outcomes before interpreting values such as 0.8 as empirical frequency.

Retraining is executed as a persistent, single-flight backend job rather than inside the website request. The Models page shows queued/running/completed history, per-machine eligibility, and readable held-out and labelled-simulation scores. These scores are advisory: a technically complete shadow remains available for an administrator to promote even when a score needs review. Confirmed-normal candidates are staged against that shadow but are consumed only when it is promoted; deleting the shadow releases them. Only one shadow may await a decision at a time. Operational failures use the configurable retry cooldown, and new candidates, material policy, or a deployed retraining-protocol code change create a new attempt signature. Every model remains an immutable bundle under `artifacts/versions/<version_id>`. Initial training writes and registers that version directory directly; the artifact root is not a staging copy. Promotion validates bundle completeness and atomically replaces only `artifacts/active.json`; it never copies model files into the artifact root. The worker, backfill, diagnostics, retraining, and website model status resolve that same pointer. PostgreSQL retains lifecycle metadata, editable display names, validation evidence, and audit history, but it is not the runtime model selector. Renaming a model changes only its display name, never its technical ID, directory, prediction references, or active pointer.

### Human-labelled accuracy simulation

The fleet-level **Accuracy simulation** page replaces the old anomaly-risk validation-test manager. An administrator chooses any registered immutable model version, adds one or more historical event ranges, and labels the status known to be true throughout each range (`OK`, `WARN`, `CRITICAL`, `STOPPED`, `STARTING`, `SENSOR_FAULT`, `NO_PREDICTION`, or `NO_DATA`). The active version is selected by default; choosing a shadow or retired version makes it possible to compare the same reusable labels across models without changing the active production pointer. Each event can span up to seven days. Lists can be saved as model-independent reusable drafts, searched and filtered in the builder, or reconstructed from any saved run. One saved list can be designated as the model validation suite. Initial training and every shadow retrain replay that independent suite and store its accuracy, coverage, timing, and mismatch evidence as an advisory score; it does not block activation or promotion. Drafts, progress, event evidence, reports, and the suite designation are stored together in `"ML".simulation_runs`. A model version referenced by a saved run cannot be deleted until those runs are removed, keeping the report-to-artifact identity intact.

Each event receives a fresh machine-specific production monitor. Context preceding `event start − 7 days` learns that machine's motion regimes and warms rolling features, condition smoothing, and trend state. Every source row from the start of that seven-day lead-in through the event end is processed in timestamp order through the same production components used by `worker.py`; no later row is available. Rolling-feature creation, machine normalization, and Isolation Forest scoring are vectorized in bounded batches, while motion transitions, smoothing, forecasting, and policy decisions remain chronological. This improves replay throughput without sampling or changing causal results. Simulation output is never inserted into `spindle_predictions`, never creates an alert, and never becomes retraining evidence.

While a run is active, its current case stores a throttled progress heartbeat inside the existing `simulation_runs.cases` JSON document. The website reports the event and phase, timeline percentage, processed rows, throughput, elapsed time, event ETA, latest source timestamp, and heartbeat age. No extra progress table is created, and progress-write failures do not invalidate an otherwise correct read-only evaluation.

Event accuracy compares the dominant causal production state inside each labelled range with the human label and requires that expected state to cover at least 50% of the event's elapsed time. Coverage is time-weighted so a short burst of dense rows cannot outweigh a longer part of the event. Mean labelled-status coverage is reported alongside event accuracy so a brief correct output cannot hide a mostly incorrect range. Lead-time compliance remains separate: a known CRITICAL event should receive a planning WARN before onset, should receive CRITICAL inside the urgent window or event, and must not receive CRITICAL prematurely; WARN events must not escalate to CRITICAL; OK events treat any WARN or CRITICAL in the lead-in or event as a false alert. The report retains the event-status confusion matrix, per-machine results, first WARN/CRITICAL timestamps and lead hours, timing exceptions, event and lead-in distributions, prediction coverage, representative evidence, a sampled condition trace, forecast evidence, and top feature contributors. A range with missing or stale source evidence is evaluated as `NO_DATA` rather than silently excluded.

Mock mode intentionally does not fabricate an accuracy report. The page explains that production mode is required, and the mock API rejects simulation creation while preserving the same endpoint contract.

For an older source table without a machine column, leave `PG_COL_MACHINE_ID` unset; all rows are assigned to `DEFAULT_MACHINE_ID`. Omitting both `--all-machines` and `--machine-id` also preserves single-machine training with `DEFAULT_MACHINE_ID`:

```powershell
python train_isolation_forest.py --source live --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full
```

## Important separation

`-Mock` is for UI/control-plane testing only. It deliberately bypasses the trained model and PostgreSQL worker. Never use mock output for maintenance decisions.

Production PostgreSQL uses `PG_PORT` from `.env`. Uvicorn's `--port 8000` is isolated from database argument parsing, so the API server port can no longer be mistaken for the PostgreSQL port.
