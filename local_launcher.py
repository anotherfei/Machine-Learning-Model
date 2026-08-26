"""Local process launcher for the spindle condition-monitoring web application.

Production mode starts FastAPI + the supervised ML worker + Vite and expects
PostgreSQL from `.env`. Mock mode starts a SQLite-backed demo FastAPI app +
Vite and deliberately skips PostgreSQL and the production ML worker.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
import socket
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
FRONTEND = ROOT / "frontend"
FRONTEND_PORT = 5173
WORKER_RESTART_CODE = 75
CATCHUP_RESTART_CODE = 76
WORKER_RESTART_DELAY_SECONDS = 1.0


def _publish_catchup_launch(days: int) -> None:
    """Make startup progress visible before importing or starting the worker."""
    path=ROOT/"artifacts"/"runtime"/"backfill_status.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp")
    payload={
        "run_id":uuid.uuid4().hex,
        "status":"launching",
        "progress":0.0,
        "days":days,
        "started_at":datetime.now(timezone.utc).isoformat(),
        "updated_at":datetime.now(timezone.utc).isoformat(),
        "message":"Starting the production worker for historical catch-up.",
    }
    try:
        temporary.write_text(json.dumps(payload),encoding="utf-8")
        os.replace(temporary,path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _npm_command() -> str | None:
    return shutil.which("npm.cmd") or shutil.which("npm")


def _port_available(host: str, port: int) -> bool:
    """Refuse to let Vite silently move to a different, misleading port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((host, port))
        except OSError:
            return False
    return True


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
    parser.add_argument(
        "--backfill-days",
        type=int,
        default=0,
        help="sequentially catch up this many source-data days before continuing realtime inference",
    )
    args = parser.parse_args()

    if not 0 <= args.backfill_days <= 3650:
        parser.error("--backfill-days must be between 0 and 3650")
    if args.mock and args.backfill_days:
        parser.error("--backfill-days is available only in production mode")

    if not args.mock and not (ROOT / ".env").exists():
        print("ERROR: .env is missing. Configure PostgreSQL or run start_project.ps1 -Mock.", file=sys.stderr)
        return 2
    if not args.no_frontend and not _port_available("127.0.0.1", FRONTEND_PORT):
        print(
            f"ERROR: frontend port {FRONTEND_PORT} is already in use. "
            "Stop the previous start_project.ps1 session before starting another one; "
            "the launcher will not serve a different project instance on another port.",
            file=sys.stderr,
        )
        return 2

    processes: list[tuple[str, subprocess.Popen]] = []
    try:
        api_module = "Demo.mock_main:app" if args.mock else "api.main:app"
        api_proc = _start("API", [sys.executable, "-m", "uvicorn", api_module, "--host", "127.0.0.1", "--port", str(args.api_port)], ROOT)
        processes.append(api_proc)
        _wait_for_api(args.api_port, api_proc[1])
        if not args.mock:
            worker_command=[sys.executable,"worker.py","--schema-ready"]
            if args.backfill_days:
                _publish_catchup_launch(args.backfill_days)
                worker_command.extend(("--catch-up-days",str(args.backfill_days)))
            processes.append(_start("worker",worker_command,ROOT))

        if not args.no_frontend:
            npm = _npm_command()
            if npm is None: raise RuntimeError("npm was not found. Install Node.js or run with -NoFrontend.")
            if not (FRONTEND / "node_modules").exists(): raise RuntimeError("frontend/node_modules is missing. Run setup_local.ps1 first.")
            if not (FRONTEND / "node_modules" / "@phosphor-icons" / "react").exists():
                raise RuntimeError("Frontend dependencies changed. Run `npm install` inside the frontend directory.")
            env = os.environ.copy(); env["SPINDLE_API_PORT"] = str(args.api_port)
            processes.append(_start("frontend", [npm, "run", "dev"], FRONTEND, env))

        mode = "MOCK DEMO (SQLite; production worker disabled)" if args.mock else "PRODUCTION (PostgreSQL + ML worker)"
        print(f"\nSpindle Condition Monitoring is running locally.\nMode: {mode}\nFrontend: http://127.0.0.1:{FRONTEND_PORT}\nAPI docs: http://127.0.0.1:{args.api_port}/docs\nPress Ctrl+C to stop all processes.\n")
        while True:
            time.sleep(1)
            for index, (name, proc) in enumerate(processes):
                code = proc.poll()
                if code is not None:
                    if name == "worker" and code == WORKER_RESTART_CODE:
                        print("[local] .env changed; restarting worker with fresh settings")
                        time.sleep(WORKER_RESTART_DELAY_SECONDS)
                        processes[index] = _start(
                            "worker",[sys.executable,"worker.py","--schema-ready"],ROOT,
                        )
                        continue
                    if name == "worker" and code == CATCHUP_RESTART_CODE:
                        print("[local] model or inference context changed; restarting sequential catch-up")
                        time.sleep(WORKER_RESTART_DELAY_SECONDS)
                        processes[index] = _start("worker",worker_command,ROOT)
                        continue
                    if name == "worker":
                        print(
                            f"[local] worker exited with code {code}; API and frontend remain "
                            "available so the failure can be inspected"
                        )
                        processes.pop(index)
                        break
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
