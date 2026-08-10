"""Versioned model-bundle registry and atomic active-version switching."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
import config

BUNDLES_DIR = os.path.join(config.ARTIFACTS_DIR, "versions")
BUNDLE_FILES = ("isolation_forest.pkl", "feature_columns.json", "calibration.json", "metadata.json", "reference_timestamps.json")


def new_version_id(now: dt.datetime | None = None) -> str:
    now = now or dt.datetime.now(dt.timezone.utc)
    return "v" + now.strftime("%Y-%m-%d-%H%M%S")


def bundle_path(version_id: str) -> str:
    return os.path.join(BUNDLES_DIR, version_id)


def reference_signature(timestamps) -> str:
    payload = "\n".join(sorted(str(x) for x in timestamps)).encode()
    return hashlib.sha256(payload).hexdigest()


def snapshot_current(version_id: str) -> str:
    target = Path(bundle_path(version_id))
    target.mkdir(parents=True, exist_ok=False)
    for name in BUNDLE_FILES:
        src = Path(config.ARTIFACTS_DIR) / name
        if src.exists():
            shutil.copy2(src, target / name)
    meta = target / "metadata.json"
    if meta.exists():
        data = json.loads(meta.read_text())
        data["version_id"] = version_id
        meta.write_text(json.dumps(data, indent=2))
    return str(target)


def install_bundle(version_id: str) -> None:
    src_dir = Path(bundle_path(version_id))
    if not src_dir.exists():
        raise FileNotFoundError(f"Missing bundle {src_dir}")
    required = BUNDLE_FILES[:4]
    missing = [name for name in required if not (src_dir / name).exists()]
    if missing:
        raise ValueError(f"Bundle {version_id} missing required files: {missing}")
    os.makedirs(config.ARTIFACTS_DIR, exist_ok=True)
    for name in BUNDLE_FILES:
        src = src_dir / name
        if src.exists():
            tmp = Path(config.ARTIFACTS_DIR) / (name + ".tmp")
            shutil.copy2(src, tmp)
            os.replace(tmp, Path(config.ARTIFACTS_DIR) / name)


def active_version(conn) -> str | None:
    with conn.cursor() as cur:
        cur.execute("SELECT version_id FROM model_versions WHERE status='active' LIMIT 1")
        row = cur.fetchone()
    return row[0] if row else None


def promote(conn, version_id: str, promoted_by: str) -> None:
    install_bundle(version_id)
    with conn.cursor() as cur:
        cur.execute("UPDATE model_versions SET status='retired' WHERE status='active' AND version_id<>%s", (version_id,))
        cur.execute("""UPDATE model_versions SET status='active', promoted_at=now(), promoted_by=%s
                       WHERE version_id=%s AND status IN ('shadow','retired','active')""", (promoted_by, version_id))
        if cur.rowcount != 1:
            raise ValueError(f"Unknown or rejected model version: {version_id}")
        cur.execute("NOTIFY model_changed, %s", (version_id,))
    conn.commit()
