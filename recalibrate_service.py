"""
Web-triggered recalibration: the same health%-anchor math as recalibrate.py
(see that module's docstring for WHY recalibration exists — the tree
generalizes across units, the baseline_mean/std anchor doesn't), sourced
from a recent window of the live production table instead of an operator-
supplied CSV path, and stored as a row in model_calibrations rather than
written straight to disk.

That last difference matters: recalibrate.py's CLI immediately overwrites
calibration_local.json the moment it's computed. A web request triggering
that same immediate overwrite would mean clicking "recalibrate" silently
changes what's deployed, with no review step and no way back except
recalibrating again. Here, computing a recalibration (POST .../recalibrate)
only stores it; a separate, explicit action
(POST .../calibration/activate, see model_registry.set_calibration) is what
actually makes it live — so "recalibrated vs normal" is a reversible choice
a human makes on the Models page, not a side effect of running the numbers.

SOURCES — two different answers to "what counts as normal?":

  "spec_bounds" (default, matches the original recalibrate.py behavior):
  a row is normal if every raw sensor reading is within config.SPEC_MAX's
  fixed rated bounds (preprocessing.select_spec_normal_rows). Static,
  independent of anything the model or the maintenance policy has ever
  said about the machine.

  "threshold": a row is normal if the live pipeline's own prediction for
  that tick was maintenance_level='OK' — i.e. health_state stayed above
  the runtime-editable MAINTENANCE_HEALTH_INSPECT/FAILURE_HEALTH_THRESHOLD
  thresholds (see Thresholds page / runtime_config.py) and the forecast
  stayed under MAINTENANCE_PROB_PLAN. This tracks whatever the operator
  currently has the maintenance policy tuned to, and needs no SPEC_MAX
  values to be filled in — useful when those spec bounds are still
  placeholders (see config.py's SPEC_MAX docstring) or when "normal"
  should mean "the policy didn't flag it" rather than "within a fixed
  sensor range".

  Both still feature-engineer the full contiguous window first and only
  filter which already-computed feature rows to keep, by timestamp —
  never the reverse (see select_spec_normal_rows()'s docstring for why).
"""
from __future__ import annotations

import json

import pandas as pd

import artifact_utils
import config
import db
import feature_engineering
import model_registry
import preprocessing

SOURCES = ("spec_bounds", "threshold")


def run(conn, version_id: str, hours: float, min_rows: int, created_by: str | None,
        source: str = "spec_bounds") -> dict:
    if source not in SOURCES:
        raise ValueError(f"source must be one of {SOURCES}, got {source!r}.")

    active = model_registry.active_version(conn)
    if active != version_id:
        raise ValueError(
            f"Only the active model version can be recalibrated from here "
            f"(active is {active!r}). Promote {version_id!r} first if you "
            f"want to recalibrate it — recalibration is a health%-anchor "
            f"override for whatever's actually deployed, and only one "
            f"bundle is ever installed at a time."
        )

    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=hours)
    table = db.get_table_name()
    rows = db.fetch_rows_between(conn, table, start.to_pydatetime(), end.to_pydatetime())
    if not rows:
        raise ValueError(f"No live rows found in the last {hours:g} hour(s) to recalibrate from.")

    dbcols = db.get_db_columns()
    rename = {dbcols["timestamp"]: config.COL_TIMESTAMP}
    rename.update({db_name: cfg_name for cfg_name, db_name in dbcols["by_config_name"].items()})
    raw_df = pd.DataFrame(rows).rename(columns=rename)
    raw_df = preprocessing.clean_data(raw_df)

    # use_local_calibration=False: always recalibrate FROM the pooled
    # default, never from whatever local override happens to already be
    # active — otherwise re-running this would compound drift onto a
    # previous recalibration instead of measuring fresh against the
    # deployed tree's own reference distribution.
    scorer, feature_cols, _meta = artifact_utils.load_artifacts(use_local_calibration=False)
    old_mean, old_std = scorer.baseline_mean, scorer.baseline_std

    # Same ordering as recalibrate.py: feature-engineer the full cleaned
    # window first, THEN select which of those already-computed feature
    # rows are normal, by timestamp — never the reverse (corrupts the
    # rolling-window features around any filtered-out gap).
    featured_df = feature_engineering.create_features(raw_df, verbose=False)

    if source == "threshold":
        ok_timestamps = db.fetch_ok_prediction_timestamps(conn, start.to_pydatetime(), end.to_pydatetime())
        if not ok_timestamps:
            raise ValueError(
                f"No maintenance_level='OK' predictions found in the last {hours:g} "
                f"hour(s) to recalibrate from. Widen the window, or use the spec-bounds "
                f"source instead."
            )
        is_normal = featured_df[config.COL_TIMESTAMP].isin(pd.to_datetime(ok_timestamps, utc=True))
        reason = "maintenance_level='OK' (runtime threshold)"
    else:
        normal_raw = preprocessing.select_spec_normal_rows(raw_df)
        is_normal = featured_df[config.COL_TIMESTAMP].isin(normal_raw[config.COL_TIMESTAMP])
        reason = "within config.SPEC_MAX bounds"

    local_features = featured_df[is_normal].reset_index(drop=True)

    if len(local_features) < min_rows:
        raise ValueError(
            f"Only {len(local_features)} normal feature rows ({reason}) in the last "
            f"{hours:g} hour(s) (need >= {min_rows}). Widen the window, lower the "
            f"minimum row count, or try the other reference source."
        )

    scorer.calibrate(local_features[feature_cols])
    calibration = scorer.calibration()

    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO model_calibrations(version_id,calibration,source_rows,source_description,created_by)
               VALUES(%s,%s::jsonb,%s,%s,%s) RETURNING id,created_at""",
            (version_id, json.dumps(calibration), len(local_features),
             f"Live window {start.isoformat()} to {end.isoformat()} ({hours:g}h), {reason}", created_by),
        )
        calibration_id, created_at = cur.fetchone()
    conn.commit()

    return {
        "calibration_id": calibration_id,
        "version_id": version_id,
        "created_at": created_at.isoformat() if hasattr(created_at, "isoformat") else created_at,
        "rows_used": len(local_features),
        "window_hours": hours,
        "source": source,
        "baseline_mean_before": old_mean,
        "baseline_std_before": old_std,
        "baseline_mean_after": scorer.baseline_mean,
        "baseline_std_after": scorer.baseline_std,
    }
