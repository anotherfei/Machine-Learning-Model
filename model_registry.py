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
        # install_bundle() only copies the pooled/default BUNDLE_FILES —
        # calibration_local.json (a recalibration override, see
        # artifact_utils.py) lives outside that set and survives a promote
        # untouched. Left alone, promoting model B while model A's
        # recalibration override is still on disk would silently apply A's
        # health%-anchor to B. Sync it to whatever THIS version's own
        # active_calibration_id says (its own choice, persisted from a
        # previous /calibration/activate call, or none) instead.
        cur.execute("SELECT active_calibration_id FROM model_versions WHERE version_id=%s", (version_id,))
        row = cur.fetchone()
        cal_id = row[0] if row else None
        if cal_id:
            cur.execute("SELECT calibration FROM model_calibrations WHERE id=%s", (cal_id,))
            cal_row = cur.fetchone()
            calibration = cal_row[0] if cal_row else None
        else:
            calibration = None
        if calibration is not None:
            artifact_utils.write_local_calibration(calibration)
        else:
            artifact_utils.clear_local_calibration()
        cur.execute("NOTIFY model_changed, %s", (version_id,))
    conn.commit()


def set_calibration(conn, version_id: str, calibration_id: int | None) -> None:
    """
    Point version_id's active_calibration_id at calibration_id (or clear it
    for None => "normal"/pooled). If version_id is the currently-active
    model, also syncs the on-disk override immediately (NOTIFY model_changed
    so the worker picks it up on its next tick) — otherwise this is just
    recorded for the next time this version gets promoted (see promote()).
    """
    with conn.cursor() as cur:
        if calibration_id is not None:
            cur.execute("SELECT id, calibration FROM model_calibrations WHERE id=%s AND version_id=%s", (calibration_id, version_id))
            row = cur.fetchone()
            if not row:
                raise ValueError(f"Calibration {calibration_id} does not belong to model version {version_id}")
            calibration = row[1]
        else:
            calibration = None
        cur.execute("UPDATE model_versions SET active_calibration_id=%s WHERE version_id=%s", (calibration_id, version_id))
        if cur.rowcount != 1:
            raise ValueError(f"Unknown model version: {version_id}")
        cur.execute("SELECT version_id FROM model_versions WHERE status='active' LIMIT 1")
        active_row = cur.fetchone()
        is_active = bool(active_row and active_row[0] == version_id)
        if is_active:
            if calibration is not None:
                artifact_utils.write_local_calibration(calibration)
            else:
                artifact_utils.clear_local_calibration()
            cur.execute("NOTIFY model_changed, %s", (version_id,))
    conn.commit()


def delete_calibration(conn, version_id: str, calibration_id: int) -> None:
    """
    Deletes one recalibration run belonging to version_id. Refuses when
    calibration_id is that version's active_calibration_id — same guard
    delete_version() uses for the active model itself: activate a
    different calibration (or Normal) first, then delete this one.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT active_calibration_id FROM model_versions WHERE version_id=%s", (version_id,))
        row = cur.fetchone()
        if not row:
            raise ValueError(f"Unknown model version: {version_id}")
        if row[0] == calibration_id:
            raise ValueError("Cannot delete the calibration currently in use — activate a different one (or Normal) first.")
        cur.execute("DELETE FROM model_calibrations WHERE id=%s AND version_id=%s", (calibration_id, version_id))
        if cur.rowcount != 1:
            raise ValueError(f"Calibration {calibration_id} does not belong to model version {version_id}")
    conn.commit()


def delete_version(conn, version_id: str) -> None:
    """
    Deletes a non-active model bundle: the DB row (model_calibrations rows
    for it cascade automatically) plus its artifact directory. The active
    model can never be deleted here — promote a different version first —
    both because it's the model actually serving predictions, and because
    model_versions_one_active plus the active_calibration_id sync in
    promote() assume there is always exactly one active row.
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
