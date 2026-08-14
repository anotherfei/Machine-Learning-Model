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

Configure `.env` with the real PostgreSQL connection, then train and place model artifacts under `artifacts/`. For a shared model, select every confirmed-healthy machine in the commissioning window:

```powershell
python train_isolation_forest.py --source live --all-machines --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full
```

This pulls a pinned commissioning window from the same PostgreSQL table `worker.py` reads. The live trainer learns stationary and rotating vibration regimes independently for every machine, builds rolling features on each complete trajectory, and then excludes `STOPPED` and `STARTING` timestamps. `--full` trusts all remaining confirmed-running rows as healthy; omit it only after genuine `SPEC_MAX` bounds have been configured. Each machine contributes the same number of rows. The newest balanced 20% (at least 20 rows per machine) is saved as validation evidence and excluded from normalizer fitting, model fitting, and condition calibration. Each remaining engineered feature is converted to a robust machine-relative value `(feature - healthy median) / healthy IQR-scale`, and one Isolation Forest is fitted to the balanced relative-feature pool. The raw fit and validation features remain stored separately in physical units for audit and future retraining. The trainer also creates a robust automatic condition-score anchor for each machine from fit rows only.

Preview the exact normalization, operating-state counts, thresholds, and balanced row counts without fitting or writing artifacts:

```powershell
python train_isolation_forest.py --source live --all-machines --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full --preview-reference
```

If a selected window contains only running data and therefore has no separable stationary cluster, `--include-non-running` is the explicit manual override. Do not use that override on a mixed production window.

`--all-machines` discovers IDs through `PG_COL_MACHINE_ID`. To train on an explicit subset instead, repeat `--machine-id`:

```powershell
python train_isolation_forest.py --source live --machine-id MACHINE-001 --machine-id MACHINE-002 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full
```

The selected period's running portions must be confirmed healthy for every included machine. The script intentionally does not invent a date range. Set `REFERENCE_WINDOW_START`/`REFERENCE_WINDOW_END` in `config.py` or pass `--start`/`--end`. Live range queries are uncapped by default; this prevents dense source windows from being silently truncated at the former 200,000-row limit. See `architecture-notes.md` → "Initial model training" for the offline-CSV fallback (`--source csv`).

### PostgreSQL index required for long training windows

Live training filters each machine by `mach_id`, restricts the selected `datetime` range, and reads it in chronological order. Large source tables therefore need a composite index with the machine column first and timestamp second. For the default production mapping, run [`database/create_training_source_index.sql`](database/create_training_source_index.sql) once using DBeaver, `psql`, or the database administration tool. Its `CREATE INDEX CONCURRENTLY` statement is designed not to block normal sensor ingestion, but it must be run outside an explicit transaction and should still be scheduled with the database administrator because a large index build consumes storage and server resources. If your `.env` maps different table or column names, change the SQL identifiers to match.

The final read-only `EXPLAIN` in that file should show `p1_sel5_vibration_mach_id_datetime_idx` (or another equivalent composite index), rather than a sequential scan. A training process still showing only `Querying PostgreSQL...` is waiting on database I/O; neither Isolation Forest CPU parallelism nor a GPU can accelerate that phase. It is safe to cancel with `Ctrl+C` because model artifacts are written only after loading, feature engineering, balancing, and fitting have completed.

Then run:

```powershell
.\start_project.ps1
```

Production mode starts:

1. FastAPI on `127.0.0.1:8000`
2. the supervised local worker process
3. React/Vite on `localhost:5173`

The worker owns the actual `SpindleMonitor` inference path. The frontend does not duplicate ML or maintenance logic.

The production dashboard discovers machine IDs and polls the latest sensor row directly from the configured PostgreSQL source. This keeps real sensor channels visible while the ML worker warms up or reconnects. Prediction WebSocket messages add model health and maintenance results when available. Source discovery failures are shown explicitly in the UI instead of silently displaying a fake/default machine.

The web console separates scope in its navigation. **Home**, Models & retraining, Global thresholds, and Environment are fleet-wide. Home includes a seven-day fleet recap built from hourly average condition scores, with a separate color-coded line and legend entry for every current source machine; machines are never averaged together. Selecting a machine opens that machine's own Overview, Status review, and History workspace. The machine selector no longer appears as a global top-bar filter, and every settings screen states whether a change applies fleet-wide or to one machine. Global thresholds exposes only runtime-writable policy: condition-score sensitivity and warm-up, condition/probability boundaries, real-time trend lookback and confidence, trend stabilization/confirmation/recovery, source polling, shutdown/restart confirmation, source-staleness timeout, and near-miss review analysis. Every control has an info icon describing its purpose and the effect of raising or lowering it; model-bound feature and normalization settings are intentionally absent. Settings that change the condition scale or buffer sizes reset in-memory monitors at a tick boundary so old and new policy state is never blended.

On worker startup, only a recent warm-up tail is read for each machine; the worker then continues from the current source watermark. It does not replay the entire historical table before reaching live data.

### Multiple machines

The sensor source may contain multiple machines in one table. Set these values in `.env`:

```text
PG_COL_MACHINE_ID=machine_id
DEFAULT_MACHINE_ID=MACHINE-001
```

`VVB001` is the shared ifm sensor model and continues to define the sensor channels and valid ranges. It is not used as a machine ID. `PG_COL_MACHINE_ID` must point to the separate source-table column containing asset identities such as `MACHINE-001`, `LINE-A-SPINDLE-02`, or your plant's own naming scheme; the website discovers its selector options from those values.

Source machine IDs are displayed without surrounding whitespace, while database filtering keeps direct equality predicates so PostgreSQL can use normal machine/timestamp indexes on large source tables. When switching between Demo and production, the frontend clears the previous session's machine list and waits for `/api/machines` before starting live polling.

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

The displayed percentage is a relative **condition score**, not measured physical remaining health. Automatic machine anchors use the healthy score median and a robust MAD/IQR spread, with a small minimum spread so nearly constant reference scores cannot amplify numerical noise. Manual web recalibration was intentionally removed: a current model selecting its own `OK` rows is circular and can silently shift maintenance decisions.

Shadow retraining follows the same shared-model rule. Confirmed-normal alerts are grouped and deduplicated in each machine's relative feature space, new rows replace older rows without allowing one machine to dominate, and fresh robust normalizers are fitted on the balanced training side only. On the first upgraded retrain, a fixed 20% machine-balanced holdout (at least 20 rows per machine) is reserved from previously accepted baseline rows and excluded from the shadow's normalization, fitting, and calibration; every new reviewed candidate stays on the training side. The bootstrap active model may have seen those legacy rows, so that first comparison is deliberately conservative. After the first promotion, the holdout is preserved outside the active and future shadow fit sets. False-positive validation is always evaluated per machine so a good fleet average cannot hide a regression on one machine. Automatically generated near-miss regression checks score the exact flagged prediction timestamp, while manually entered regression windows retain peak-within-window semantics. A promotable shadow bundles its fit reference, held-out validation evidence, fresh normalizers, and automatic condition anchors so the complete context switches atomically with the model.

Historical backfill keeps predictions and Status Review alerts consistent in the same transaction. Pending machine-generated alerts are updated, created, or removed to match the replayed decision. Predictions tied to an already reviewed alert or near-miss are not rewritten, preserving the exact evidence behind the human decision. The live API also withholds condition and maintenance values whenever the newest PostgreSQL sensor row is newer than the worker's prediction, and the frontend shows that the worker is catching up instead of combining mismatched timestamps.

Forecast maintenance risk uses the Brownian first-passage probability of crossing the critical condition boundary at any time within the selected horizon. Its diffusion scale comes from detrended condition innovations and actual timestamp gaps, not from the regression residual level. This is mathematically consistent with the stated model but remains a model-estimated risk; validate its numeric calibration against real maintenance and failure outcomes before interpreting values such as 0.8 as empirical frequency.

Retraining is executed as a persistent, single-flight backend job rather than inside the website request. The Models page shows queued/running/completed history, per-machine eligibility, readable validation evidence, and an administrator-managed regression-test list. Confirmed-normal candidates are staged against a passed shadow but are consumed only when that shadow is promoted; deleting the shadow releases them. Only one shadow may await a decision at a time. An unchanged validation rejection is not automatically repeated, operational failures use the configurable retry cooldown, and new candidates, material policy/regression-test changes, or a deployed retraining-protocol code change create a new attempt signature. Promotion validates database state first, serializes competing promotions, installs the bundle, activates its machine-specific normalizers and condition anchors, and consumes the staged candidates in the same database transaction.

For an older source table without a machine column, leave `PG_COL_MACHINE_ID` unset; all rows are assigned to `DEFAULT_MACHINE_ID`. Omitting both `--all-machines` and `--machine-id` also preserves single-machine training with `DEFAULT_MACHINE_ID`:

```powershell
python train_isolation_forest.py --source live --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z --full
```

## Important separation

`-Mock` is for UI/control-plane testing only. It deliberately bypasses the trained model and PostgreSQL worker. Never use mock output for maintenance decisions.

Production PostgreSQL uses `PG_PORT` from `.env`. Uvicorn's `--port 8000` is isolated from database argument parsing, so the API server port can no longer be mistaken for the PostgreSQL port.
