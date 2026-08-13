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

Mock mode creates `mock_demo.db` in the project root. It contains seeded users, sensor/prediction history, pending alerts, model versions, thresholds, and mock environment settings. The mock API also emits a synthetic live tick once per second through the same `/ws/live` endpoint used by the frontend.

Reset the temporary database:

```powershell
.\reset_mock.ps1
```

Then start again with `-Mock`.

## Production mode

Configure `.env` with the real PostgreSQL connection, then train and place model artifacts under `artifacts/`:

```powershell
python train_isolation_forest.py
```

By default this fits the model on a pinned commissioning window pulled from the same PostgreSQL table `worker.py` reads in production — set `REFERENCE_WINDOW_START`/`REFERENCE_WINDOW_END` in `config.py` (or pass `--start`/`--end`) first. See `architecture-notes.md` → "Initial model training" for the full picture, including the offline-CSV fallback (`--source csv`).

Then run:

```powershell
.\start_project.ps1
```

Production mode starts:

1. FastAPI on `127.0.0.1:8000`
2. the supervised local worker process
3. React/Vite on `localhost:5173`

The worker owns the actual `SpindleMonitor` inference path. The frontend does not duplicate ML or maintenance logic.

### Multiple machines

The sensor source may contain multiple machines in one table. Set these values in `.env`:

```text
PG_COL_MACHINE_ID=machine_id
DEFAULT_MACHINE_ID=VVB001
```

Every `(timestamp, machine_id)` row is polled in order. The worker creates an independent rolling feature window, Kalman filter, trend history, and maintenance debouncer for each machine, while all machines use the active model bundle. Predictions, alerts, near-miss calculations, history, and live WebSocket messages are isolated by `machine_id`.

Health-anchor recalibrations are also machine-specific: activating a recalibration for one machine does not change another machine's scorer.

For an older source table without a machine column, leave `PG_COL_MACHINE_ID` unset; all rows are assigned to `DEFAULT_MACHINE_ID`. Live commissioning training targets one unit at a time with `--machine-id`:

```powershell
python train_isolation_forest.py --source live --machine-id VVB002 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
```

## Important separation

`-Mock` is for UI/control-plane testing only. It deliberately bypasses the trained model and PostgreSQL worker. Never use mock output for maintenance decisions.

Production PostgreSQL uses `PG_PORT` from `.env`. Uvicorn's `--port 8000` is isolated from database argument parsing, so the API server port can no longer be mistaken for the PostgreSQL port.
