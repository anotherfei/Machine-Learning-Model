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
  identified a confirmed-healthy commissioning stretch in the live data;
  `artifact_utils.save_artifacts()` then records the exact
  `reference_timestamps` used, so the fitted window stays reproducible
  and auditable after the fact even though the source table isn't static.
- **`"csv"`** — the original offline-file behavior, reading
  `config.RAW_DATA_PATH`. Kept for offline experimentation and for CI/test
  fixtures that shouldn't need a reachable database.

Either way, `preprocessing.select_spec_normal_rows()` (or `--full` to skip
it) decides which rows within that window actually get fit on — that part
of the pipeline is unchanged by where the window's rows come from.

`recalibrate.py` is a separate, later step: re-anchoring the health%
scale for one specific deployment without refitting the tree. It still
matters even under `REFERENCE_SOURCE="live"` the moment a tree trained on
one unit's commissioning data gets reused for a different unit.

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
