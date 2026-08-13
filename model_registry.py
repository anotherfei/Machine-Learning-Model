"""Versioned model-bundle registry and atomic active-version switching."""
from __future__ import annotations
import datetime as dt
import hashlib
import json
import os
import shutil
from pathlib import Path
import artifact_utils
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
        # The worker applies database-backed per-machine calibrations. A
        # legacy global file must not survive and leak across machines.
        artifact_utils.clear_local_calibration()
        cur.execute("NOTIFY model_changed, %s", (version_id,))
    conn.commit()


def set_calibration(conn, version_id: str, calibration_id: int | None, machine_id: str) -> None:
    """Activate a calibration for one machine, or restore its pooled baseline."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM model_versions WHERE version_id=%s",(version_id,))
        if not cur.fetchone():
            raise ValueError(f"Unknown model version: {version_id}")
        if calibration_id is not None:
            cur.execute("SELECT id FROM model_calibrations WHERE id=%s AND version_id=%s AND machine_id=%s",(calibration_id,version_id,machine_id))
            if not cur.fetchone():
                raise ValueError(f"Calibration {calibration_id} does not belong to machine {machine_id} and model version {version_id}")
            cur.execute("""INSERT INTO machine_model_calibrations(machine_id,version_id,calibration_id)
                           VALUES(%s,%s,%s) ON CONFLICT(machine_id,version_id) DO UPDATE
                           SET calibration_id=EXCLUDED.calibration_id,updated_at=now()""",(machine_id,version_id,calibration_id))
        else:
            cur.execute("DELETE FROM machine_model_calibrations WHERE machine_id=%s AND version_id=%s",(machine_id,version_id))
        cur.execute("SELECT 1 FROM model_versions WHERE version_id=%s AND status='active'",(version_id,))
        if cur.fetchone():
            cur.execute("NOTIFY model_changed, %s",(version_id,))
    conn.commit()


def machine_calibration(conn, version_id: str, machine_id: str):
    """Return the active health-anchor calibration for one machine."""
    with conn.cursor() as cur:
        cur.execute("""SELECT mc.calibration FROM machine_model_calibrations mmc
                       JOIN model_calibrations mc ON mc.id=mmc.calibration_id
                       WHERE mmc.machine_id=%s AND mmc.version_id=%s""",(machine_id,version_id))
        row=cur.fetchone()
    return row[0] if row else None


def delete_calibration(conn, version_id: str, calibration_id: int, machine_id: str) -> None:
    """Delete one machine calibration unless that machine is using it."""
    with conn.cursor() as cur:
        cur.execute("SELECT 1 FROM machine_model_calibrations WHERE machine_id=%s AND version_id=%s AND calibration_id=%s",(machine_id,version_id,calibration_id))
        if cur.fetchone():
            raise ValueError("Cannot delete the calibration currently in use — activate a different one (or Normal) first.")
        cur.execute("DELETE FROM model_calibrations WHERE id=%s AND version_id=%s AND machine_id=%s",(calibration_id,version_id,machine_id))
        if cur.rowcount != 1:
            raise ValueError(f"Calibration {calibration_id} does not belong to machine {machine_id} and model version {version_id}")
    conn.commit()


def delete_version(conn, version_id: str) -> None:
    """
    Deletes a non-active model bundle: the DB row (model_calibrations rows
    for it cascade automatically) plus its artifact directory. The active
    model can never be deleted here — promote a different version first —
    because it is the model currently serving predictions.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT status, artifact_path FROM model_versions WHERE version_id=%s", (version_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Unknown model version: {version_id}")
        status, artifact_path = row
        if status == "active":
            raise ValueError("Cannot delete the active model version — promote a different version first.")
        cur.execute("DELETE FROM model_versions WHERE version_id=%s", (version_id,))
    conn.commit()
    if artifact_path and os.path.isdir(artifact_path):
        shutil.rmtree(artifact_path, ignore_errors=True)
