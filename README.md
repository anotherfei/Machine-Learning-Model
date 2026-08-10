# Spindle Condition Monitoring

Unsupervised VVB001 spindle condition monitoring with a FastAPI control plane and React/TypeScript web interface.

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

Configure `.env` with the real PostgreSQL connection and place trained model artifacts under `artifacts/`, then run:

```powershell
.\start_project.ps1
```

Production mode starts:

1. FastAPI on `127.0.0.1:8000`
2. the supervised local worker process
3. React/Vite on `localhost:5173`

The worker owns the actual `SpindleMonitor` inference path. The frontend does not duplicate ML or maintenance logic.

## Important separation

`-Mock` is for UI/control-plane testing only. It deliberately bypasses the trained model and PostgreSQL worker. Never use mock output for maintenance decisions.

Production PostgreSQL uses `PG_PORT` from `.env`. Uvicorn's `--port 8000` is isolated from database argument parsing, so the API server port can no longer be mistaken for the PostgreSQL port.
