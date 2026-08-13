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

import pandas as pd

import config
import db
import preprocessing
import feature_engineering
import artifact_utils
from isolation_forest import AnomalyScorer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full", action="store_true",
        help="Treat the ENTIRE reference window as normal instead of "
             "spec-filtering it (preprocessing.select_spec_normal_rows()). "
             "Only use this when you're sure every row in the window is "
             "genuinely normal operation — with --full, nothing checks "
             "that for you; every row is trusted as-is."
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
    print(f"[train] Health over {health_check_label} — "
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
            print(f"[train] Machine {machine_id!r} local calibration health — "
                  f"min: {local_health.min():.1f}, max: {local_health.max():.1f}, "
                  f"mean: {local_health.mean():.1f}")

    artifact_utils.save_artifacts(
        scorer,
        feature_cols,
        reference_timestamps=reference_df[config.COL_TIMESTAMP],
        reference_rows=reference_rows,
        machine_calibrations=machine_calibrations,
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
    rename = {dbcols["timestamp"]: config.COL_TIMESTAMP}
    rename.update({db_name: cfg_name for cfg_name, db_name in dbcols["by_config_name"].items()})

    raw_df = pd.DataFrame(rows).rename(columns=rename)
    raw_df = preprocessing.clean_data(raw_df)
    print(f"[train] Pulled {len(raw_df)} live rows for machine {machine_id!r} from {table!r} "
          f"between {start} and {end} (after cleaning).")
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


def _select_live_reference(raw_df, featured_df, use_full, machine_id):
    if use_full:
        reference_df = featured_df.reset_index(drop=True)
        print(f"[train] Machine {machine_id!r}: --full uses all "
              f"{len(reference_df)} feature rows (spec filter skipped).")
    else:
        # Filter raw rows first, but compute rolling features on the complete
        # per-machine trajectory so gaps never corrupt a rolling window.
        normal_raw = preprocessing.select_spec_normal_rows(raw_df)
        is_reference = featured_df[config.COL_TIMESTAMP].isin(normal_raw[config.COL_TIMESTAMP])
        reference_df = featured_df[is_reference].reset_index(drop=True)
        print(f"[train] Machine {machine_id!r}: {len(reference_df)}/{len(featured_df)} "
              f"feature rows pass the spec filter.")

    if reference_df.empty:
        raise ValueError(
            f"Machine {machine_id!r} has no usable healthy feature rows in the "
            "selected commissioning window. Widen the window, verify the spec "
            "bounds, or use --full only if the entire window is confirmed healthy."
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
        if len(frame) > rows_per_machine:
            selected = frame.sample(n=rows_per_machine, random_state=42)
            selected = selected.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
        else:
            selected = frame.reset_index(drop=True)
        balanced[machine_id] = selected

    print(f"[train] Balanced pool: {rows_per_machine} healthy feature rows per "
          f"machine, {rows_per_machine * len(balanced)} total. Available before "
          f"balancing: {counts}.")
    return balanced, counts, rows_per_machine


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

        machine_health_frames[machine_id] = featured_df
        machine_reference_frames[machine_id] = _select_live_reference(
            raw_df, featured_df, args.full, machine_id
        )

    balanced_frames, available_counts, rows_per_machine = _balanced_reference_pool(
        machine_reference_frames, args.max_rows_per_machine
    )
    reference_df = pd.concat(balanced_frames.values(), ignore_index=True)
    health_check_df = pd.concat(machine_health_frames.values(), ignore_index=True)
    reference_rows = [
        {"machine_id": machine_id, "timestamp": timestamp}
        for machine_id, frame in balanced_frames.items()
        for timestamp in frame[config.COL_TIMESTAMP]
    ]

    _fit_and_save(
        reference_df,
        feature_cols,
        health_check_df=health_check_df,
        health_check_label="all selected commissioning windows",
        reference_rows=reference_rows,
        machine_reference_frames=machine_reference_frames,
        machine_health_frames=machine_health_frames,
        metadata_extra={
            "training_mode": "balanced_pooled" if len(machine_ids) > 1 else "single_machine",
            "training_machine_ids": machine_ids,
            "available_reference_rows_per_machine": available_counts,
            "model_fit_rows_per_machine": rows_per_machine,
            "balance_method": "equal_rows_deterministic_sample",
            "reference_window_start": start.isoformat(),
            "reference_window_end": end.isoformat(),
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

    _fit_and_save(
        reference_df, feature_cols,
        health_check_df=df,
        health_check_label="full trajectory",
    )


def main():
    args = parse_args()
    source = args.source or config.REFERENCE_SOURCE
    print(f"[train] REFERENCE_SOURCE = {source!r}"
          f"{' (overridden via --source)' if args.source else ' (from config.py)'}")

    if source == "live":
        _train_from_live(args)
    elif source == "csv":
        _train_from_csv(args)
    else:
        raise ValueError(f"Unknown REFERENCE_SOURCE {source!r} — must be 'live' or 'csv'.")


if __name__ == "__main__":
    main()
