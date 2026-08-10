# Architecture notes

## Production

`React/Vite -> FastAPI -> PostgreSQL` for control-plane requests and history.

The production worker runs separately and owns `SpindleMonitor`, feature engineering, anomaly scoring, health estimation, forecasting, and maintenance recommendation. Results are written to `spindle_predictions` and streamed to the UI.

## Temporary mock mode

`React/Vite -> FastAPI mock API -> SQLite mock_demo.db`.

The production worker is not started. The mock API exposes the same frontend-facing endpoints and generates synthetic live VVB001 ticks so the web UI can be verified without a PostgreSQL server or trained artifacts.

## Local ports

- Frontend: `5173`
- FastAPI: `8000`
- PostgreSQL production default: `5432`

Database configuration is read from `.env`. Server command-line flags are not parsed as database overrides.
