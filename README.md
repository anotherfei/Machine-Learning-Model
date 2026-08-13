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

Configure `.env` with the real PostgreSQL connection, then train and place model artifacts under `artifacts/`. For a shared model, select every confirmed-healthy machine in the commissioning window:

```powershell
python train_isolation_forest.py --source live --all-machines --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
```

This pulls a pinned commissioning window from the same PostgreSQL table `worker.py` reads. Features are built independently for every machine, each machine contributes the same number of healthy feature rows, and one Isolation Forest is fitted to the combined balanced pool. The trainer also creates a separate initial health calibration for each machine. On the first production startup, the API registers the shared model and activates those machine-specific calibrations automatically.

`--all-machines` discovers IDs through `PG_COL_MACHINE_ID`. To train on an explicit subset instead, repeat `--machine-id`:

```powershell
python train_isolation_forest.py --source live --machine-id MACHINE-001 --machine-id MACHINE-002 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
```

The selected period must be confirmed healthy for every included machine. The script intentionally does not invent a date range. Set `REFERENCE_WINDOW_START`/`REFERENCE_WINDOW_END` in `config.py` or pass `--start`/`--end`. See `architecture-notes.md` → "Initial model training" for the offline-CSV fallback (`--source csv`).

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
DEFAULT_MACHINE_ID=MACHINE-001
```

`VVB001` is the shared ifm sensor model and continues to define the sensor channels and valid ranges. It is not used as a machine ID. `PG_COL_MACHINE_ID` must point to the separate source-table column containing asset identities such as `MACHINE-001`, `LINE-A-SPINDLE-02`, or your plant's own naming scheme; the website discovers its selector options from those values.

Every `(timestamp, machine_id)` row is polled in order. The worker creates an independent rolling feature window, Kalman filter, trend history, and maintenance debouncer for each machine, while all machines use the active model bundle. Predictions, alerts, near-miss calculations, history, and live WebSocket messages are isolated by `machine_id`.

Health-anchor recalibrations are also machine-specific: activating a recalibration for one machine does not change another machine's scorer.

Shadow retraining follows the same shared-model rule. Confirmed-normal alerts are grouped and deduplicated by machine, new rows replace older rows without allowing one machine to dominate, and the proposed reference is balanced before fitting. False-positive validation runs separately for every machine, while regression windows are rebuilt and tested against their recorded machine ID. A promotable shadow includes fresh per-machine calibrations that take effect with the model when it is promoted.

For an older source table without a machine column, leave `PG_COL_MACHINE_ID` unset; all rows are assigned to `DEFAULT_MACHINE_ID`. Omitting both `--all-machines` and `--machine-id` also preserves single-machine training with `DEFAULT_MACHINE_ID`:

```powershell
python train_isolation_forest.py --source live --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
```

## Important separation

`-Mock` is for UI/control-plane testing only. It deliberately bypasses the trained model and PostgreSQL worker. Never use mock output for maintenance decisions.

Production PostgreSQL uses `PG_PORT` from `.env`. Uvicorn's `--port 8000` is isolated from database argument parsing, so the API server port can no longer be mistaken for the PostgreSQL port.
