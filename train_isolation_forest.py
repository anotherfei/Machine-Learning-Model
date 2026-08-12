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
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Shared: fit + save, given a feature table and a reference-row selection
# ---------------------------------------------------------------------------
def _fit_and_save(reference_df, feature_cols, health_check_df, health_check_label):
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

    artifact_utils.save_artifacts(scorer, feature_cols, reference_timestamps=reference_df[config.COL_TIMESTAMP])


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


def _load_live_raw(start, end):
    conn = db.get_connection()
    try:
        table = db.get_table_name()
        # .to_pydatetime(): same conversion backfill.py uses before handing
        # a pd.Timestamp to psycopg2, so tz-aware/naive comparisons against
        # the DB column behave the same way here as they do there.
        rows = db.fetch_rows_between(conn, table, start.to_pydatetime(), end.to_pydatetime())
    finally:
        conn.close()

    if not rows:
        raise ValueError(
            f"No rows found in table {table!r} between {start} and {end}. "
            f"Check REFERENCE_WINDOW_START/END and the PG_* connection "
            f"settings in .env."
        )

    dbcols = db.get_db_columns()
    rename = {dbcols["timestamp"]: config.COL_TIMESTAMP}
    rename.update({db_name: cfg_name for cfg_name, db_name in dbcols["by_config_name"].items()})

    raw_df = pd.DataFrame(rows).rename(columns=rename)
    raw_df = preprocessing.clean_data(raw_df)
    print(f"[train] Pulled {len(raw_df)} live rows from {table!r} "
          f"between {start} and {end} (after cleaning).")
    return raw_df


def _train_from_live(args):
    start, end = _resolve_live_window(args)
    raw_df = _load_live_raw(start, end)

    featured_df = feature_engineering.create_features(raw_df)
    feature_cols = feature_engineering.get_feature_columns(featured_df)

    if args.full:
        # Trust the whole commissioning window as normal — no spec filter.
        reference_df = featured_df.reset_index(drop=True)
        print(f"[train] --full: using all {len(reference_df)} live rows as "
              f"the reference set (spec filter skipped).")
    else:
        # Spec-based row selection needs the raw sensor values, which don't
        # survive feature_engineering.create_features() — filter on raw_df
        # (already cleaned above), then map that selection onto featured_df
        # by timestamp.
        normal_raw = preprocessing.select_spec_normal_rows(raw_df)
        is_reference = featured_df[config.COL_TIMESTAMP].isin(normal_raw[config.COL_TIMESTAMP])
        reference_df = featured_df[is_reference].reset_index(drop=True)
        print(f"[train] Spec-based reference set: {len(reference_df)}/{len(featured_df)} "
              f"live rows pass the spec filter.")

    # There's no larger "full trajectory" to sanity-check against here —
    # unlike the CSV path, the live pull IS the commissioning window, not
    # a longer file that also contains later degradation. So this checks
    # the reference window scoring itself: health should sit high/flat,
    # since by definition this is the data the model calls "normal".
    _fit_and_save(
        reference_df, feature_cols,
        health_check_df=featured_df,
        health_check_label="the commissioning window itself (expect high/flat — "
                            "this isn't a healthy-to-failed trajectory, just the reference data)",
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
