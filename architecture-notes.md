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
  `--machine-id` to select an explicit subset. Rolling features are built
  separately per machine and healthy feature rows are sampled down to the
  same count per machine before they are combined. This prevents the
  highest-volume machine from dominating the one shared Isolation Forest.
  Sampling is deterministic (`random_state=42`) so rerunning against the
  same pinned source rows produces the same balanced reference selection.
  `artifact_utils.save_artifacts()` records machine-aware reference rows,
  machine counts, and the exact training selection for auditability.
- **`"csv"`** — the original offline-file behavior, reading
  `config.RAW_DATA_PATH`. Kept for offline experimentation and for CI/test
  fixtures that shouldn't need a reachable database.

Either way, `preprocessing.select_spec_normal_rows()` (or `--full` to skip
it) decides which rows within that window are accepted as healthy. Only use
`--full` when the entire window is independently confirmed healthy.

After the shared tree is fitted, the live trainer separately calibrates its
health-percentage anchor against each machine's complete healthy reference
set. Those calibrations are bundled with the model and imported into
`machine_model_calibrations` when production registers its first model, so
machines share the anomaly model but do not share a health baseline.

`recalibrate.py` is a separate, later step: re-anchoring the health%
scale for one specific machine without refitting the tree. It is useful
when a new machine joins after initial training or a serviced machine's
healthy operating baseline has materially changed.

## Production

`React/Vite -> FastAPI -> PostgreSQL` for control-plane requests and history.

The production worker runs separately and owns `SpindleMonitor`, feature engineering, anomaly scoring, health estimation, forecasting, and maintenance recommendation. Results are written to `spindle_predictions` and streamed to the UI.

## Temporary mock mode

`React/Vite -> FastAPI mock API -> SQLite mock_demo.db`.

The production worker is not started. The mock API exposes the same frontend-facing endpoints and generates synthetic live ticks for multiple machine IDs so the web UI can be verified without a PostgreSQL server or trained artifacts.

## Local ports

- Frontend: `5173`
- FastAPI: `8000`
- PostgreSQL production default: `5432`

Database configuration is read from `.env`. Server command-line flags are not parsed as database overrides.
