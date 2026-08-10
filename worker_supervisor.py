"""Local supervisor for the production spindle-monitoring worker.

worker.py exits with code 75 after a saved .env change. This supervisor starts
a fresh Python process so environment values and DB connections are rebuilt.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESTART_CODE = 75
RESTART_DELAY_SECONDS = 1.0


def main() -> int:
    print("Spindle Condition Monitoring local worker supervisor started")
    while True:
        proc = subprocess.Popen([sys.executable, "worker.py"], cwd=ROOT, env=os.environ.copy())
        try:
            code = proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            return 0

        if code == RESTART_CODE:
            print("[worker-supervisor] .env changed; restarting worker with fresh settings")
            time.sleep(RESTART_DELAY_SECONDS)
            continue

        if code == 0:
            print("[worker-supervisor] worker exited normally; supervisor stopping")
            return 0

        print(f"[worker-supervisor] worker exited with code {code}; not restarting automatically")
        return code


if __name__ == "__main__":
    raise SystemExit(main())
