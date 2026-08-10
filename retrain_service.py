"""Human-confirmed-normal shadow retraining and pre-deploy validation gate."""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

import artifact_utils
import config
import model_registry
import runtime_config
from isolation_forest import AnomalyScorer


def _current_reference_features(feature_cols: list[str]) -> pd.DataFrame:
    if not os.path.exists(config.FEATURES_DATA_PATH):
        return pd.DataFrame(columns=feature_cols)
    df = pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])
    ts = artifact_utils.load_reference_timestamps()
    if ts is None:
        return pd.DataFrame(columns=feature_cols)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.DateOffset(months=int(runtime_config.get("REFERENCE_WINDOW_MONTHS", 6)))
    ts_set = set(pd.to_datetime(ts, utc=True))
    dfts = pd.to_datetime(df[config.COL_TIMESTAMP], utc=True)
    mask = dfts.isin(ts_set) & (dfts >= cutoff)
    return df.loc[mask, feature_cols].reset_index(drop=True)


def _candidate_rows(conn, feature_cols: list[str]):
    months = int(runtime_config.get("REFERENCE_WINDOW_MONTHS", 6))
    with conn.cursor() as cur:
        cur.execute("""SELECT rc.id, rc.tick_timestamp, a.feature_vector
                       FROM reference_candidates rc JOIN alerts a ON a.id=rc.alert_id
                       WHERE rc.added_to_reference_at IS NULL
                         AND rc.tick_timestamp >= now() - (%s || ' months')::interval
                         AND a.status='confirmed_normal'
                       ORDER BY rc.tick_timestamp""", (months,))
        rows = cur.fetchall()
    out = []
    for cid, ts, vec in rows:
        if not vec:
            continue
        if isinstance(vec, str): vec = json.loads(vec)
        if all(c in vec for c in feature_cols):
            out.append((cid, ts, np.array([float(vec[c]) for c in feature_cols], dtype=float)))
    return out


def _dedup(rows):
    if not rows:
        return []
    threshold = float(runtime_config.get("REFERENCE_COSINE_SIMILARITY", 0.98))
    hours = float(runtime_config.get("REFERENCE_DEDUP_WINDOW_HOURS", 24))
    accepted = []
    for item in rows:
        cid, ts, vec = item
        duplicate = False
        for _, ats, avec in reversed(accepted):
            if abs((ts - ats).total_seconds()) > hours * 3600:
                break
            denom = np.linalg.norm(vec) * np.linalg.norm(avec)
            similarity = float(np.dot(vec, avec) / denom) if denom else 1.0
            if similarity >= threshold:
                duplicate = True
                break
        if not duplicate:
            accepted.append(item)
    return accepted


def _risk(scorer, X):
    if len(X) == 0: return np.array([])
    score = scorer.score(X)
    health = scorer.health_from_score(score)
    return np.clip(1.0 - health / 100.0, 0.0, 1.0)


def _regression_feature_windows(conn, feature_cols):
    if not os.path.exists(config.FEATURES_DATA_PATH):
        return []
    df = pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])
    with conn.cursor() as cur:
        cur.execute("SELECT id, lower(timestamp_range), upper(timestamp_range), minimum_anomaly_risk FROM regression_tests ORDER BY id")
        tests = cur.fetchall()
    out=[]
    for rid, lo, hi, minimum in tests:
        ts = pd.to_datetime(df[config.COL_TIMESTAMP], utc=True)
        mask = (ts >= pd.Timestamp(lo)) & (ts <= pd.Timestamp(hi))
        out.append((rid, df.loc[mask, feature_cols], float(minimum)))
    return out


def should_retrain(conn) -> tuple[bool, dict]:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*), min(created_at) FROM reference_candidates WHERE added_to_reference_at IS NULL")
        count, oldest = cur.fetchone()
    batch = int(runtime_config.get("RETRAIN_BATCH_SIZE", 50))
    cap = int(runtime_config.get("RETRAIN_TIME_CAP_DAYS", 30))
    age_days = (dt.datetime.now(dt.timezone.utc)-oldest).total_seconds()/86400 if oldest else 0
    return bool(count >= batch or (count > 0 and age_days >= cap)), {"pending": count, "batch_size": batch, "oldest_age_days": age_days, "time_cap_days": cap}


def run_shadow_retrain(conn, force: bool=False) -> dict:
    due, status = should_retrain(conn)
    if not force and not due:
        return {"started": False, **status}
    active_scorer, feature_cols, active_meta = artifact_utils.load_artifacts()
    base = _current_reference_features(feature_cols)
    candidate_rows = _dedup(_candidate_rows(conn, feature_cols))
    if not candidate_rows:
        raise ValueError("No eligible, non-duplicate confirmed-normal candidates are available")
    candidate_df = pd.DataFrame([x[2] for x in candidate_rows], columns=feature_cols)
    reference = pd.concat([base, candidate_df], ignore_index=True).drop_duplicates()
    if len(reference) < max(20, len(feature_cols)+1):
        raise ValueError(f"Reference set too small after pruning/dedup: {len(reference)} rows")

    shadow = AnomalyScorer().fit(reference)
    # Gate 1: confirmed-normal/reference false alarms cannot materially regress.
    health_cut = float(runtime_config.get("MAINTENANCE_HEALTH_INSPECT", config.MAINTENANCE_HEALTH_INSPECT))
    active_health = active_scorer.health_from_score(active_scorer.score(reference))
    shadow_health = shadow.health_from_score(shadow.score(reference))
    active_fp = float(np.mean(active_health <= health_cut))
    shadow_fp = float(np.mean(shadow_health <= health_cut))
    fp_pass = shadow_fp <= active_fp + 0.02

    # Gate 2: every permanent confirmed-FN regression window must remain caught.
    regression=[]
    regression_pass=True
    for rid, X, minimum in _regression_feature_windows(conn, feature_cols):
        if X.empty:
            regression.append({"id": rid, "pass": False, "reason": "no matching feature rows"})
            regression_pass=False
            continue
        max_risk = float(np.max(_risk(shadow, X)))
        passed = max_risk >= minimum
        regression.append({"id": rid, "max_anomaly_risk": max_risk, "minimum": minimum, "pass": passed})
        regression_pass &= passed

    report = {"reference_rows": len(reference), "new_candidates": len(candidate_rows), "active_reference_fp": active_fp,
              "shadow_reference_fp": shadow_fp, "reference_fp_gate_pass": fp_pass,
              "regression_tests": regression, "regression_gate_pass": regression_pass}
    accepted = fp_pass and regression_pass
    version_id = model_registry.new_version_id()
    target = Path(model_registry.bundle_path(version_id)); target.mkdir(parents=True, exist_ok=False)
    joblib.dump(shadow.model, target/"isolation_forest.pkl")
    (target/"feature_columns.json").write_text(json.dumps(feature_cols, indent=2))
    (target/"calibration.json").write_text(json.dumps(shadow.calibration(), indent=2))
    metadata = dict(active_meta); metadata.update({"version_id": version_id, "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(), "n_features": len(feature_cols), "pipeline_hash": artifact_utils.config_hash(), "human_confirmed_retrain": True})
    (target/"metadata.json").write_text(json.dumps(metadata, indent=2))
    signature = model_registry.reference_signature([x[1] for x in candidate_rows])
    with conn.cursor() as cur:
        cur.execute("INSERT INTO model_versions(version_id,artifact_path,reference_signature,status,validation_report) VALUES(%s,%s,%s,%s,%s::jsonb)",
                    (version_id, str(target), signature, "shadow" if accepted else "rejected", json.dumps(report)))
        if accepted:
            ids=[x[0] for x in candidate_rows]
            cur.execute("UPDATE reference_candidates SET added_to_reference_at=now() WHERE id=ANY(%s)", (ids,))
    conn.commit()
    return {"started": True, "version_id": version_id, "status": "shadow" if accepted else "rejected", "validation": report}
