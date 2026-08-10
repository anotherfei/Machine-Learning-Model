# Temporary mock web demo

This mode is only for checking that the web application works. It does not use the production PostgreSQL database or production ML worker.

## First-time setup

From PowerShell in the extracted project folder:

```powershell
.\setup_local.ps1
```

## Start the mock demo

```powershell
.\start_project.ps1 -Mock
```

The launcher now waits for the API health endpoint before starting Vite. You should see an `API ready` message before the frontend starts.

Open:

- Web UI: `http://localhost:5173`
- API health: `http://localhost:8000/api/health`
- FastAPI docs: `http://localhost:8000/docs`

Default local-demo login:

- Username: `admin`
- Password: `change-me-on-first-deployment`

## Reset synthetic data

Stop the project with Ctrl+C, then run:

```powershell
.\reset_mock.ps1
.\start_project.ps1 -Mock
```

## If the page does not load

The frontend no longer intentionally displays a blank page while authentication is being checked. It will show either a loading state, login screen, backend warning, or an explicit frontend error.

If needed, verify `http://localhost:8000/api/health` first. It should return JSON with `"ok": true`.
