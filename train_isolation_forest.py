"""
Fit the Isolation Forest on the reference/commissioning baseline window
and save artifacts. No labels anywhere in this script.

Where the reference window comes from is controlled by
config.REFERENCE_SOURCE:

  "live" (default) - pulls directly from the same Postgres table +
      connection worker.py/db.py use in production, restricted to
      [REFERENCE_WINDOW_START, REFERENCE_WINDOW_END). Training and
      production then share one source of truth instead of two schemas
      that can drift apart. Requires those two timestamps to be set
      (in config.py, or via --start/--end below) — this script will not
      guess a window from "whatever the table currently holds", because
      that would make every run pull a different, unreproducible
      reference set.
  "csv" - config.RAW_DATA_PATH, the original offline-file behavior. Kept
      for offline experimentation / CI fixtures that shouldn't need a
      reachable database.

Usage:
    python train_isolation_forest.py
    python train_isolation_forest.py --full
    python train_isolation_forest.py --source live --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
    python train_isolation_forest.py --source live --all-machines --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
    python train_isolation_forest.py --source live --machine-id MACHINE-001 --machine-id MACHINE-002 --start 2026-01-05T00:00:00Z --end 2026-01-07T00:00:00Z
    python train_isolation_forest.py --source csv
"""

import argparse
import time

import numpy as np
import pandas as pd

import config
import db
import preprocessing
import feature_engineering
import artifact_utils
import operating_state
import machine_normalization
from isolation_forest import AnomalyScorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Treat every confirmed-RUNNING row in the reference window as "
             "normal instead of "
             "spec-filtering it (preprocessing.select_spec_normal_rows()). "
             "Only use this when you're sure the running portions are "
             "genuinely healthy. STOPPED and STARTING rows remain excluded "
             "unless --include-non-running is also supplied."
    )
    parser.add_argument(
        "--include-non-running", action="store_true",
        help="Explicit unsafe override that disables the sensor-derived "
             "RUNNING gate for live training. Use only when the selected "
             "window is independently confirmed to contain running data only."
    )
    parser.add_argument(
        "--source", choices=["live", "csv"], default=None,
        help="Override config.REFERENCE_SOURCE for this run without "
             "editing config.py."
    )
    parser.add_argument(
        "--start", default=None,
        help="Override config.REFERENCE_WINDOW_START for this run "
             "(only used when the resolved source is 'live'). ISO 8601, "
             "e.g. 2026-01-05T00:00:00Z."
    )
    parser.add_argument(
        "--end", default=None,
        help="Override config.REFERENCE_WINDOW_END for this run "
             "(only used when the resolved source is 'live')."
    )
    machine_group = parser.add_mutually_exclusive_group()
    machine_group.add_argument(
        "--machine-id", action="append", default=None,
        help="Machine to include in live training. Repeat this option (or use "
             "a comma-separated value) to build one balanced shared model from "
             "multiple machines. Defaults to DEFAULT_MACHINE_ID."
    )
    machine_group.add_argument(
        "--all-machines", action="store_true",
        help="Discover every machine ID in the live source and include all of "
             "them in the balanced shared model."
    )
    parser.add_argument(
        "--max-rows-per-machine", type=int, default=None,
        help="Optional upper bound on healthy feature rows sampled from each "
             "machine. All machines still contribute exactly the same number."
    )
    parser.add_argument(
        "--preview-reference", action="store_true",
        help="Load, normalize, feature-engineer, operating-state filter, and "
             "balance the live reference selection, then stop before fitting "
             "or writing model artifacts."
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared: fit + save, given a feature table and a reference-row selection
# ---------------------------------------------------------------------------
def _fit_and_save(
    reference_df,
    feature_cols,
    health_check_df,
    health_check_label,
    reference_rows=None,
    machine_reference_frames=None,
    machine_health_frames=None,
    machine_feature_normalizers=None,
    reference_features=None,
    validation_features=None,
    metadata_extra=None,
):
    scorer = AnomalyScorer()
    scorer.fit(reference_df[feature_cols])
    print(f"[train] Calibration: {scorer.calibration()}")

    # Sanity check — not a labeled evaluation (no labels exist in this
    # codebase); just a distributional check that health looks reasonable
    # rather than sitting flat at 100 or crashing to 0 everywhere.
    scores = scorer.score(health_check_df[feature_cols])
    health = scorer.health_from_score(scores)
    print(f"[train] Condition score over {health_check_label} — "
          f"min: {health.min():.1f}, max: {health.max():.1f}, "
          f"mean: {health.mean():.1f}")

    machine_calibrations = None
    if machine_reference_frames:
        machine_calibrations = {}
        for machine_id, machine_reference in machine_reference_frames.items():
            local_scorer = AnomalyScorer()
            local_scorer.model = scorer.model
            local_scorer.calibrate(machine_reference[feature_cols])
            machine_calibrations[machine_id] = {
                "calibration": local_scorer.calibration(),
                "source_rows": len(machine_reference),
            }

            machine_health = (machine_health_frames or {}).get(machine_id, machine_reference)
            local_scores = local_scorer.score(machine_health[feature_cols])
            local_health = local_scorer.health_from_score(local_scores)
            print(f"[train] Machine {machine_id!r} automatic condition anchor — "
                  f"min: {local_health.min():.1f}, max: {local_health.max():.1f}, "
                  f"mean: {local_health.mean():.1f}")

    artifact_utils.save_artifacts(
        scorer,
        feature_cols,
        reference_timestamps=reference_df[config.COL_TIMESTAMP],
        reference_rows=reference_rows,
        reference_features=reference_features,
        validation_features=validation_features,
        machine_calibrations=machine_calibrations,
        machine_feature_normalizers=machine_feature_normalizers,
        metadata_extra=metadata_extra,
    )


# ---------------------------------------------------------------------------
# Live-source path: reference window = an explicit, pinned range pulled
# from the production Postgres table itself.
# ---------------------------------------------------------------------------
def _resolve_live_window(args):
    start = args.start or config.REFERENCE_WINDOW_START
    end = args.end or config.REFERENCE_WINDOW_END
    if not start or not end:
        raise ValueError(
            "REFERENCE_SOURCE is 'live' but no commissioning window is set. "
            "Set config.REFERENCE_WINDOW_START and config.REFERENCE_WINDOW_END "
            "to a confirmed-healthy stretch of the live table (or pass "
            "--start/--end on the command line) before training. See "
            "config.py's REFERENCE_WINDOW_START comment for why this isn't "
            "inferred automatically."
        )
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    if end <= start:
        raise ValueError(f"REFERENCE_WINDOW_END ({end}) must be after REFERENCE_WINDOW_START ({start}).")
    return start, end


def _load_live_raw(start, end, machine_id):
    print(
        f"[train] Querying PostgreSQL for machine {machine_id!r} between "
        f"{start} and {end}...",
        flush=True,
    )
    print(
        "[train] If this query remains here for more than about a minute, "
        "cancel with Ctrl+C and install the composite source index documented "
        "in database/create_training_source_index.sql.",
        flush=True,
    )
    query_started = time.perf_counter()
    conn = db.get_connection()
    try:
        table = db.get_table_name()
        # .to_pydatetime(): same conversion backfill.py uses before handing
        # a pd.Timestamp to psycopg2, so tz-aware/naive comparisons against
        # the DB column behave the same way here as they do there.
        rows = db.fetch_rows_between(conn, table, start.to_pydatetime(), end.to_pydatetime(), machine_id=machine_id)
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"No rows found for machine {machine_id!r} in table {table!r} between {start} and {end}. "
            f"Check REFERENCE_WINDOW_START/END and the PG_* connection "
            f"settings in .env."
        )

    dbcols = db.get_db_columns()
    raw_df = db.canonical_sensor_frame(rows, dbcols)
    raw_df = preprocessing.clean_data(raw_df)
    print(f"[train] Pulled {len(raw_df)} live rows for machine {machine_id!r} from {table!r} "
          f"between {start} and {end} in {time.perf_counter()-query_started:.1f}s "
          "(after cleaning).")
    return raw_df


def _resolve_live_machine_ids(args):
    if args.all_machines:
        conn = db.get_connection()
        try:
            machine_ids = db.fetch_machine_ids(conn, db.get_table_name())
        finally:
            conn.close()
    elif args.machine_id:
        machine_ids = []
        for value in args.machine_id:
            machine_ids.extend(part.strip() for part in value.split(","))
    else:
        machine_ids = [db.DEFAULT_MACHINE_ID]

    # Preserve command/source order while removing blanks and duplicates.
    machine_ids = list(dict.fromkeys(machine_id for machine_id in machine_ids if machine_id))
    if not machine_ids:
        raise ValueError("No machine IDs were selected for live training.")
    print(f"[train] Training machines ({len(machine_ids)}): {', '.join(machine_ids)}")
    return machine_ids


def _running_feature_rows(raw_df, featured_df, machine_id, include_non_running=False):
    """Select model-applicable rows without breaking contiguous windows."""
    if include_non_running:
        print(f"[train] Machine {machine_id!r}: --include-non-running keeps all "
              f"{len(featured_df)} complete feature rows.")
        return featured_df.reset_index(drop=True)

    detector = operating_state.OperatingStateDetector(history_rows=len(raw_df))
    readings = raw_df[config.RAW_SENSOR_COLS].to_dict("records")
    for reading in readings:
        detector.observe_history(reading)
    if not detector.fit_history():
        raise ValueError(
            f"Machine {machine_id!r} has no clearly separable stationary and "
            "rotating regimes in the selected window. Choose a window containing "
            "both regimes, or use --include-non-running only if the whole window "
            "is independently confirmed to contain running data."
        )

    running_timestamps = []
    state_counts = {state: 0 for state in operating_state.VALID_STATES}
    for timestamp, reading in zip(raw_df[config.COL_TIMESTAMP], readings):
        state = detector.update(reading)
        state_counts[state.state] = state_counts.get(state.state, 0) + 1
        if state.state == "RUNNING" and not state.low_motion:
            running_timestamps.append(timestamp)

    selected = featured_df[
        featured_df[config.COL_TIMESTAMP].isin(running_timestamps)
    ].reset_index(drop=True)
    print(
        f"[train] Machine {machine_id!r}: operating-state filter kept "
        f"{len(selected)}/{len(featured_df)} feature rows; states={state_counts}; "
        f"activity thresholds stop={detector.stop_threshold:.6g}, "
        f"run={detector.run_threshold:.6g}."
    )
    if selected.empty:
        raise ValueError(
            f"Machine {machine_id!r} produced no confirmed-RUNNING feature rows."
        )
    return selected


def _select_live_reference(raw_df, applicable_df, use_full, machine_id):
    if use_full:
        reference_df = applicable_df.reset_index(drop=True)
        print(f"[train] Machine {machine_id!r}: --full uses all "
              f"{len(reference_df)} confirmed-applicable feature rows "
              "(spec filter skipped).")
    else:
        # Filter raw rows first, but compute rolling features on the complete
        # per-machine trajectory so gaps never corrupt a rolling window.
        normal_raw = preprocessing.select_spec_normal_rows(raw_df)
        is_reference = applicable_df[config.COL_TIMESTAMP].isin(normal_raw[config.COL_TIMESTAMP])
        reference_df = applicable_df[is_reference].reset_index(drop=True)
        print(f"[train] Machine {machine_id!r}: {len(reference_df)}/{len(applicable_df)} "
              f"applicable feature rows pass the spec filter.")

    if reference_df.empty:
        raise ValueError(
            f"Machine {machine_id!r} has no usable healthy feature rows in the "
            "selected commissioning window. Widen the window, verify the spec "
            "bounds, or use --full only if every selected running row is confirmed healthy."
        )
    return reference_df


def _balanced_reference_pool(machine_reference_frames, max_rows_per_machine=None):
    counts = {machine_id: len(frame) for machine_id, frame in machine_reference_frames.items()}
    rows_per_machine = min(counts.values())
    if max_rows_per_machine is not None:
        if max_rows_per_machine < 1:
            raise ValueError("--max-rows-per-machine must be at least 1.")
        rows_per_machine = min(rows_per_machine, max_rows_per_machine)

    balanced = {}
    for machine_id, frame in machine_reference_frames.items():
        ordered = frame.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        if len(frame) > rows_per_machine:
            holdout_rows = min(
                rows_per_machine,
                max(
                    artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS,
                    int(np.ceil(rows_per_machine * artifact_utils.VALIDATION_HOLDOUT_FRACTION)),
                ),
            )
            newest = ordered.tail(holdout_rows)
            fit_rows = rows_per_machine - holdout_rows
            fit_pool = ordered.iloc[:-holdout_rows]
            sampled_fit = fit_pool.sample(n=fit_rows, random_state=42)
            selected = pd.concat([sampled_fit, newest], ignore_index=True)
            selected = selected.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        else:
            selected = ordered
        balanced[machine_id] = selected

    print(f"[train] Balanced pool before validation holdout: {rows_per_machine} "
          f"healthy feature rows per machine, {rows_per_machine * len(balanced)} "
          f"total. Available before balancing: {counts}.")
    return balanced, counts, rows_per_machine


def _split_validation_holdout(balanced_frames, feature_cols):
    """Create equal, forward-looking per-machine commissioning holdouts."""
    fraction = artifact_utils.VALIDATION_HOLDOUT_FRACTION
    minimum_holdout = artifact_utils.VALIDATION_HOLDOUT_MIN_ROWS
    minimum_fit = max(20, len(feature_cols) + 1)
    training_frames = {}
    validation_frames = {}
    for machine_id, frame in balanced_frames.items():
        ordered = frame.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        holdout_rows = max(minimum_holdout, int(np.ceil(len(ordered) * fraction)))
        if len(ordered) < holdout_rows + minimum_fit:
            raise ValueError(
                f"Machine {machine_id!r} needs at least {holdout_rows + minimum_fit} "
                f"balanced healthy feature rows to keep {holdout_rows} for validation "
                f"and {minimum_fit} for model fitting; only {len(ordered)} are available."
            )
        validation_frames[machine_id] = ordered.tail(holdout_rows).reset_index(drop=True)
        training_frames[machine_id] = ordered.iloc[:-holdout_rows].reset_index(drop=True)
    return training_frames, validation_frames


def _train_from_live(args):
    start, end = _resolve_live_window(args)
    machine_ids = _resolve_live_machine_ids(args)
    machine_reference_frames = {}
    machine_health_frames = {}
    feature_cols = None

    # Build rolling features independently. Combining raw rows first would
    # let windows cross machine boundaries and create impossible readings.
    for machine_id in machine_ids:
        raw_df = _load_live_raw(start, end, machine_id)
        featured_df = feature_engineering.create_features(raw_df)
        if featured_df.empty:
            raise ValueError(
                f"Machine {machine_id!r} produced no complete feature rows. "
                f"Each machine needs at least {config.MIN_PERIODS} clean source rows."
            )

        current_feature_cols = feature_engineering.get_feature_columns(featured_df)
        if feature_cols is None:
            feature_cols = current_feature_cols
        elif current_feature_cols != feature_cols:
            raise ValueError(f"Feature columns differ for machine {machine_id!r}.")

        applicable_df = _running_feature_rows(
            raw_df,
            featured_df,
            machine_id,
            include_non_running=args.include_non_running,
        )
        machine_health_frames[machine_id] = applicable_df
        machine_reference_frames[machine_id] = _select_live_reference(
            raw_df, applicable_df, args.full, machine_id
        )

    balanced_frames, available_counts, rows_per_machine = _balanced_reference_pool(
        machine_reference_frames, args.max_rows_per_machine
    )
    training_frames, validation_frames = _split_validation_holdout(
        balanced_frames, feature_cols
    )
    machine_feature_normalizers = machine_normalization.fit_normalizers(
        training_frames, feature_cols
    )
    normalized_training_frames = {
        machine_id: machine_normalization.transform(
            frame, feature_cols, machine_feature_normalizers[machine_id], machine_id
        )
        for machine_id, frame in training_frames.items()
    }
    normalized_health_frames = {
        machine_id: machine_normalization.transform(
            frame, feature_cols, machine_feature_normalizers[machine_id], machine_id
        )
        for machine_id, frame in machine_health_frames.items()
    }
    reference_df = pd.concat(normalized_training_frames.values(), ignore_index=True)
    health_check_df = pd.concat(normalized_health_frames.values(), ignore_index=True)
    reference_rows = [
        {"machine_id": machine_id, "timestamp": timestamp}
        for machine_id, frame in training_frames.items()
        for timestamp in frame[config.COL_TIMESTAMP]
    ]
    reference_features = pd.concat(
        [frame.assign(machine_id=machine_id) for machine_id, frame in training_frames.items()],
        ignore_index=True,
    )[["machine_id", config.COL_TIMESTAMP, *feature_cols]]
    validation_features = pd.concat(
        [frame.assign(machine_id=machine_id) for machine_id, frame in validation_frames.items()],
        ignore_index=True,
    )[["machine_id", config.COL_TIMESTAMP, *feature_cols]]

    if args.preview_reference:
        print(
            "[train] Reference preview complete; no model was fitted and no "
            "artifacts were written."
        )
        return

    _fit_and_save(
        reference_df,
        feature_cols,
        health_check_df=health_check_df,
        health_check_label="all selected commissioning windows",
        reference_rows=reference_rows,
        reference_features=reference_features,
        validation_features=validation_features,
        machine_reference_frames=normalized_training_frames,
        machine_health_frames=normalized_health_frames,
        machine_feature_normalizers=machine_feature_normalizers,
        metadata_extra={
            "training_mode": "balanced_pooled" if len(machine_ids) > 1 else "single_machine",
            "training_machine_ids": machine_ids,
            "available_reference_rows_per_machine": available_counts,
            "balanced_rows_per_machine_before_holdout": rows_per_machine,
            "model_fit_rows_per_machine": {
                machine_id: len(frame) for machine_id, frame in training_frames.items()
            },
            "validation_rows_per_machine": {
                machine_id: len(frame) for machine_id, frame in validation_frames.items()
            },
            "validation_holdout_fraction": artifact_utils.VALIDATION_HOLDOUT_FRACTION,
            "validation_holdout_lineage": "commissioning_forward_holdout",
            "balance_method": "equal_rows_deterministic_sample",
            "model_feature_space": machine_normalization.METHOD,
            "reference_window_start": start.isoformat(),
            "reference_window_end": end.isoformat(),
            "postgres_sensor_scales": config.POSTGRES_SENSOR_SCALES,
            "operating_state_filter": (
                "disabled_explicitly" if args.include_non_running
                else "confirmed_running_per_machine"
            ),
        },
    )


# ---------------------------------------------------------------------------
# CSV-source path: original offline-file behavior.
# ---------------------------------------------------------------------------
def _get_or_build_csv_features():
    if preprocessing.cached_features_are_stale():
        print(f"[train] Cached features missing or stale for {config.RAW_DATA_PATH} — "
              f"rebuilding from raw data (see preprocessing.cached_features_are_stale()).")
        preprocessing.run_preprocessing()
        return feature_engineering.run_feature_engineering()

    print(f"[train] Loading cached features from {config.FEATURES_DATA_PATH}")
    return pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])


def _train_from_csv(args):
    df = _get_or_build_csv_features()
    feature_cols = feature_engineering.get_feature_columns(df)

    if args.full:
        # Trust the whole file as normal — no spec filter, no reference-
        # window cutoff. Every row in config.RAW_DATA_PATH becomes the
        # reference set the Isolation Forest is fit on.
        reference_df = df.reset_index(drop=True)
        print(f"[train] --full: using all {len(reference_df)} rows as the "
              f"reference set (spec filter skipped).")
    else:
        # Spec-based row selection needs the raw sensor values, which don't
        # survive feature_engineering.create_features() — so reload/re-clean
        # the raw CSV here rather than reading them off the feature table.
        # Cheap: just load_data() + clean_data(), no feature engineering rerun.
        raw_df = preprocessing.clean_data(preprocessing.load_data())
        normal_raw = preprocessing.select_spec_normal_rows(raw_df)

        is_reference = df[config.COL_TIMESTAMP].isin(normal_raw[config.COL_TIMESTAMP])
        reference_df = df[is_reference].reset_index(drop=True)
        rest_df = df[~is_reference].reset_index(drop=True)
        print(f"[train] Spec-based reference set: {len(reference_df)} rows, "
              f"rest of trajectory: {len(rest_df)} rows.")

    machine_id = db.DEFAULT_MACHINE_ID
    training_frames, validation_frames = _split_validation_holdout(
        {machine_id: reference_df}, feature_cols
    )
    training_reference = training_frames[machine_id]
    validation_reference = validation_frames[machine_id]
    machine_feature_normalizers = machine_normalization.fit_normalizers(
        training_frames, feature_cols
    )
    normalizer = machine_feature_normalizers[machine_id]
    normalized_reference = machine_normalization.transform(
        training_reference, feature_cols, normalizer, machine_id
    )
    normalized_full = machine_normalization.transform(
        df, feature_cols, normalizer, machine_id
    )
    reference_features = training_reference.assign(machine_id=machine_id)[
        ["machine_id", config.COL_TIMESTAMP, *feature_cols]
    ]
    validation_features = validation_reference.assign(machine_id=machine_id)[
        ["machine_id", config.COL_TIMESTAMP, *feature_cols]
    ]
    reference_rows = [
        {"machine_id": machine_id, "timestamp": timestamp}
        for timestamp in training_reference[config.COL_TIMESTAMP]
    ]

    _fit_and_save(
        normalized_reference, feature_cols,
        health_check_df=normalized_full,
        health_check_label="full trajectory",
        reference_rows=reference_rows,
        reference_features=reference_features,
        validation_features=validation_features,
        machine_reference_frames={machine_id: normalized_reference},
        machine_health_frames={machine_id: normalized_full},
        machine_feature_normalizers=machine_feature_normalizers,
        metadata_extra={
            "training_mode": "single_machine_csv",
            "training_machine_ids": [machine_id],
            "model_feature_space": machine_normalization.METHOD,
            "model_fit_rows_per_machine": {machine_id: len(training_reference)},
            "validation_rows_per_machine": {machine_id: len(validation_reference)},
            "validation_holdout_fraction": artifact_utils.VALIDATION_HOLDOUT_FRACTION,
            "validation_holdout_lineage": "commissioning_forward_holdout",
        },
    )


def main():
    args = parse_args()
    source = args.source or config.REFERENCE_SOURCE
    print(f"[train] REFERENCE_SOURCE = {source!r}"
          f"{' (overridden via --source)' if args.source else ' (from config.py)'}")

    if source != "live" and (args.include_non_running or args.preview_reference):
        raise ValueError(
            "--include-non-running and --preview-reference are live-source options."
        )

    if source == "live":
        _train_from_live(args)
    elif source == "csv":
        _train_from_csv(args)
    else:
        raise ValueError(f"Unknown REFERENCE_SOURCE {source!r} — must be 'live' or 'csv'.")


if __name__ == "__main__":
    main()
