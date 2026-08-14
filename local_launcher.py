"""Local process launcher for the spindle condition-monitoring web application.

Production mode starts FastAPI + the supervised ML worker + Vite and expects
PostgreSQL from `.env`. Mock mode starts a SQLite-backed demo FastAPI app +
Vite and deliberately skips PostgreSQL and the production ML worker.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"


def _npm_command() -> str | None:
    return shutil.which("npm.cmd") or shutil.which("npm")


def _start(name: str, command: list[str], cwd: Path, env: dict[str, str] | None = None) -> tuple[str, subprocess.Popen]:
    print(f"[local] starting {name}: {' '.join(command)}")
    kwargs = {"cwd": cwd, "env": env or os.environ.copy()}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    return name, subprocess.Popen(command, **kwargs)


def _stop(processes: list[tuple[str, subprocess.Popen]]) -> None:
    for name, proc in reversed(processes):
        if proc.poll() is None:
            print(f"[local] stopping {name}")
            proc.terminate()
    deadline = time.time() + 5
    for _name, proc in processes:
        if proc.poll() is None:
            try: proc.wait(timeout=max(0.1, deadline - time.time()))
            except subprocess.TimeoutExpired: proc.kill()



def _wait_for_api(port: int, proc: subprocess.Popen, timeout: float = 20.0) -> None:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout
    last_error = "not ready"
    while time.time() < deadline:
        code = proc.poll()
        if code is not None:
            raise RuntimeError(f"API exited during startup with code {code}")
        try:
            with urllib.request.urlopen(url, timeout=1.0) as response:
                if response.status == 200:
                    print(f"[local] API ready: {url}")
                    return
        except Exception as exc:
            last_error = str(exc)
        time.sleep(0.25)
    raise RuntimeError(f"API did not become ready within {timeout:.0f}s: {last_error}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run spindle condition monitoring locally")
    parser.add_argument("--mock", action="store_true", help="use temporary SQLite demo database; do not start production worker")
    parser.add_argument("--no-frontend", action="store_true", help="start only the API (and worker in production mode)")
    parser.add_argument("--api-port", type=int, default=8000)
    args = parser.parse_args()

    if not args.mock and not (ROOT / ".env").exists():
        print("ERROR: .env is missing. Configure PostgreSQL or run start_project.ps1 -Mock.", file=sys.stderr)
        return 2

    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        api_module = "Demo.mock_main:app" if args.mock else "api.main:app"
        api_proc = _start("API", [sys.executable, "-m", "uvicorn", api_module, "--host", "127.0.0.1", "--port", str(args.api_port)], ROOT)
        processes.append(api_proc)
        _wait_for_api(args.api_port, api_proc[1])
        if not args.mock:
            processes.append(_start("worker", [sys.executable, "worker_supervisor.py"], ROOT))

        if not args.no_frontend:
            npm = _npm_command()
            if npm is None: raise RuntimeError("npm was not found. Install Node.js or run with -NoFrontend.")
            if not (FRONTEND / "node_modules").exists(): raise RuntimeError("frontend/node_modules is missing. Run setup_local.ps1 first.")
            if not (FRONTEND / "node_modules" / "@phosphor-icons" / "react").exists():
                raise RuntimeError("Frontend dependencies changed. Run `npm install` inside the frontend directory.")
            env = os.environ.copy(); env["SPINDLE_API_PORT"] = str(args.api_port)
            processes.append(_start("frontend", [npm, "run", "dev"], FRONTEND, env))

        mode = "MOCK DEMO (SQLite; production worker disabled)" if args.mock else "PRODUCTION (PostgreSQL + ML worker)"
        print(f"\nSpindle Condition Monitoring is running locally.\nMode: {mode}\nFrontend: http://localhost:5173\nAPI docs: http://localhost:{args.api_port}/docs\nPress Ctrl+C to stop all processes.\n")
        while True:
            time.sleep(1)
            for name, proc in processes:
                code = proc.poll()
                if code is not None:
                    print(f"[local] {name} exited with code {code}; shutting down the remaining processes")
                    return code
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr); return 2
    finally:
        _stop(processes)


if __name__ == "__main__":
    raise SystemExit(main())
