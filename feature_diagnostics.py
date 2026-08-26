"""Offline model-space feature diagnostic.

Run this manually after training or retraining::

    python feature_diagnostics.py

Isolation Forest does not expose conventional feature importances. This
utility estimates them by repeatedly shuffling one feature and measuring the
change in raw anomaly score. It also reports highly correlated feature pairs.

The default run uses a deterministic, machine-balanced sample so the report is
fast enough to be useful and no machine dominates because it has more rows.
Use ``--all-rows`` only when a deliberately exhaustive diagnostic is worth the
additional time and memory. This utility never changes model artifacts or the
realtime pipeline.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import shutil
import sys
import time

import numpy as np
import pandas as pd

import artifact_utils
import attribution
import machine_normalization


DEFAULT_PERMUTATIONS = 5
DEFAULT_ROWS_PER_MACHINE = 5_000
DEFAULT_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_CORRELATION_THRESHOLD = 0.95
RANDOM_STATE = 42
PROGRESS_WIDTH = 30
PROGRESS_REFRESH_SECONDS = 0.1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _correlation_threshold(value: str) -> float:
    parsed = float(value)
    if not 0.0 < parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate model feature importance and feature redundancy."
    )
    parser.add_argument(
        "--permutations",
        type=_positive_int,
        default=DEFAULT_PERMUTATIONS,
        help=f"shuffles per feature (default: {DEFAULT_PERMUTATIONS})",
    )
    parser.add_argument(
        "--rows-per-machine",
        type=_positive_int,
        default=DEFAULT_ROWS_PER_MACHINE,
        help=(
            "maximum deterministic sample per machine "
            f"(default: {DEFAULT_ROWS_PER_MACHINE:,})"
        ),
    )
    parser.add_argument(
        "--all-rows",
        action="store_true",
        help="use the complete bundled reference set instead of balanced sampling",
    )
    parser.add_argument(
        "--workers",
        type=_positive_int,
        default=DEFAULT_WORKERS,
        help=f"concurrent permutation workers (default: {DEFAULT_WORKERS})",
    )
    parser.add_argument(
        "--correlation-threshold",
        type=_correlation_threshold,
        default=DEFAULT_CORRELATION_THRESHOLD,
        help=(
            "absolute correlation reported as redundant "
            f"(default: {DEFAULT_CORRELATION_THRESHOLD})"
        ),
    )
    return parser.parse_args()


def _load_reference_features(
    feature_cols: list[str],
    rows_per_machine: int | None,
) -> tuple[pd.DataFrame, int, dict[str, int]]:
    reference_df = artifact_utils.load_reference_features()
    if reference_df is None or reference_df.empty:
        raise RuntimeError(
            "No machine-aware reference_features.csv is bundled. Retrain with the "
            "current trainer before running model-space diagnostics."
        )
    if "machine_id" not in reference_df.columns:
        raise RuntimeError(
            "The bundled reference set has no machine_id column. Retrain with the "
            "current machine-aware trainer."
        )

    total_rows = len(reference_df)
    normalizers = artifact_utils.load_machine_feature_normalizers()
    normalized_frames = []
    retained_counts: dict[str, int] = {}

    for machine_id, frame in reference_df.groupby("machine_id", sort=True):
        machine_id = str(machine_id)
        if rows_per_machine is not None and len(frame) > rows_per_machine:
            frame = frame.sample(
                n=rows_per_machine,
                random_state=RANDOM_STATE,
                replace=False,
            )
        retained_counts[machine_id] = len(frame)
        normalizer = machine_normalization.for_machine(normalizers, machine_id)
        normalized_frames.append(
            machine_normalization.transform(frame, feature_cols, normalizer, machine_id)
        )

    if not normalized_frames:
        raise RuntimeError("The bundled reference set contains no machine rows.")
    normalized = pd.concat(normalized_frames, ignore_index=True)
    return normalized, total_rows, retained_counts


def _format_duration(seconds: float | None) -> str:
    if seconds is None or not np.isfinite(seconds):
        return "--"
    seconds = max(0, int(round(seconds)))
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


class ProgressReporter:
    """Render one repainting terminal line, or sparse milestones in logs."""

    def __init__(self, total: int):
        self.total = total
        self.started_at = time.perf_counter()
        self.last_rendered_at = 0.0
        self.last_log_milestone = -1
        self.interactive = bool(getattr(sys.stdout, "isatty", lambda: False)())

    @staticmethod
    def _fill_character() -> str:
        character = "■"
        try:
            character.encode(sys.stdout.encoding or "utf-8")
        except (LookupError, UnicodeEncodeError):
            return "#"
        return character

    def _message(self, completed: int, now: float) -> str:
        progress = completed / self.total if self.total else 1.0
        elapsed = max(0.0, now - self.started_at)
        rate = completed / elapsed if completed and elapsed > 0 else 0.0
        remaining = (self.total - completed) / rate if rate > 0 else None
        rate_text = f"{rate:.1f}/s" if rate > 0 else "--/s"
        columns = max(20, shutil.get_terminal_size(fallback=(100, 24)).columns)
        prefix = "[feature_diagnostics] " if self.interactive and columns >= 90 else ""
        if columns >= 80:
            suffix = (
                f"] {progress:5.1%} {completed}/{self.total} | "
                f"{rate_text} | ETA {_format_duration(remaining)}"
            )
        elif columns >= 55:
            suffix = (
                f"] {progress:5.1%} {completed}/{self.total} | "
                f"ETA {_format_duration(remaining)}"
            )
        else:
            suffix = f"] {progress:4.0%} {completed}/{self.total}"

        available = columns - len(prefix) - len(suffix) - 1
        bar_width = max(5, min(PROGRESS_WIDTH, available))
        filled = min(bar_width, int(progress * bar_width))
        bar = self._fill_character() * filled + "-" * (bar_width - filled)
        return f"{prefix}[{bar}{suffix}"

    def update(self, completed: int, *, force: bool = False) -> None:
        now = time.perf_counter()
        if self.interactive:
            if (
                not force
                and completed not in (0, self.total)
                and now - self.last_rendered_at < PROGRESS_REFRESH_SECONDS
            ):
                return
            message = self._message(completed, now)
            # Clear the current row first so a shorter ETA cannot leave stale text.
            sys.stdout.write(f"\x1b[2K\r{message}")
            sys.stdout.flush()
            self.last_rendered_at = now
            if completed >= self.total:
                sys.stdout.write("\n")
                sys.stdout.flush()
            return

        # A redirected stream cannot repaint. Emit only 10% milestones instead
        # of writing hundreds of carriage-return records into a log file.
        progress = completed / self.total if self.total else 1.0
        milestone = min(10, int(progress * 10))
        if force or milestone > self.last_log_milestone:
            print(f"[feature_diagnostics] {self._message(completed, now)}", flush=True)
            self.last_log_milestone = milestone

    def interrupt(self) -> None:
        if self.interactive:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _permutation_delta(
    scorer,
    values: np.ndarray,
    feature_cols: list[str],
    baseline_scores: np.ndarray,
    feature_index: int,
    repeat: int,
) -> tuple[int, int, float]:
    """Score one deterministic feature shuffle in a worker thread."""
    seed = RANDOM_STATE + feature_index * 10_007 + repeat
    rng = np.random.RandomState(seed)
    shuffled_values = values.copy()
    shuffled_values[:, feature_index] = values[
        rng.permutation(len(values)), feature_index
    ]
    shuffled_frame = pd.DataFrame(
        shuffled_values,
        columns=feature_cols,
        copy=False,
    )
    shuffled_scores = scorer.score(shuffled_frame)
    delta = float(np.mean(np.abs(shuffled_scores - baseline_scores)))
    return feature_index, repeat, delta


def permutation_importance(
    scorer,
    X: pd.DataFrame,
    feature_cols: list[str],
    permutations: int,
    workers: int,
) -> pd.Series:
    """Estimate score sensitivity with concurrent deterministic shuffles."""
    values = X.loc[:, feature_cols].to_numpy(dtype=float, copy=True)
    baseline_frame = pd.DataFrame(values, columns=feature_cols, copy=False)
    baseline_scores = scorer.score(baseline_frame)
    deltas: dict[int, list[float | None]] = {
        index: [None] * permutations for index in range(len(feature_cols))
    }
    total = len(feature_cols) * permutations
    progress = ProgressReporter(total)
    progress.update(0, force=True)

    with ThreadPoolExecutor(
        max_workers=workers,
        thread_name_prefix="feature-diagnostic",
    ) as executor:
        futures = [
            executor.submit(
                _permutation_delta,
                scorer,
                values,
                feature_cols,
                baseline_scores,
                feature_index,
                repeat,
            )
            for feature_index in range(len(feature_cols))
            for repeat in range(permutations)
        ]
        try:
            for completed, future in enumerate(as_completed(futures), start=1):
                feature_index, repeat, delta = future.result()
                deltas[feature_index][repeat] = delta
                progress.update(completed, force=completed >= total)
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            progress.interrupt()
            print("[feature_diagnostics] Cancellation requested; stopping queued work...")
            raise

    return pd.Series(
        {
            feature_cols[index]: float(np.mean(np.asarray(deltas[index], dtype=float)))
            for index in range(len(feature_cols))
        },
        dtype=float,
    )


def find_redundant_pairs(
    X: pd.DataFrame,
    feature_cols: list[str],
    threshold: float,
) -> list[tuple[str, str, str, float]]:
    """Flag highly correlated pairs within each non-cross-term sensor group."""
    groups: dict[str, list[str]] = {}
    for column in feature_cols:
        if column.endswith("_corr"):
            continue
        sensor = attribution.sensor_label(column)
        groups.setdefault(sensor, []).append(column)

    redundant = []
    for sensor, columns in groups.items():
        if len(columns) < 2:
            continue
        correlation = X[columns].corr().abs()
        for index, first in enumerate(columns):
            for second in columns[index + 1:]:
                value = correlation.loc[first, second]
                if pd.notna(value) and value >= threshold:
                    redundant.append((sensor, first, second, float(value)))
    return redundant


def main() -> None:
    args = _parse_args()
    rows_per_machine = None if args.all_rows else args.rows_per_machine

    print("[feature_diagnostics] Loading model and reference set...")
    scorer, feature_cols, _metadata = artifact_utils.load_artifacts()
    reference_df, total_rows, retained_counts = _load_reference_features(
        feature_cols,
        rows_per_machine,
    )

    retained_summary = ", ".join(
        f"{machine_id}={count:,}"
        for machine_id, count in retained_counts.items()
    )
    print(
        f"[feature_diagnostics] Using {len(reference_df):,}/{total_rows:,} reference "
        f"rows across {len(retained_counts)} machines ({retained_summary})."
    )
    if args.all_rows:
        print(
            "[feature_diagnostics] Full-reference mode enabled. Runtime and temporary "
            "memory scale with the complete reference set."
        )

    score_evaluations = len(feature_cols) * args.permutations + 1
    row_scores = len(reference_df) * score_evaluations
    print(
        f"[feature_diagnostics] {len(feature_cols)} features, "
        f"{args.permutations} permutations/feature, {args.workers} workers; "
        f"{score_evaluations:,} score evaluations "
        f"(~{row_scores:,} row scores)."
    )
    importances = permutation_importance(
        scorer,
        reference_df,
        feature_cols,
        args.permutations,
        args.workers,
    )

    sensor_labels = [attribution.sensor_label(column) for column in importances.index]
    by_sensor = importances.groupby(sensor_labels).sum()
    importance_total = float(by_sensor.sum())
    if importance_total > 0:
        by_sensor_percent = 100 * by_sensor / importance_total
    else:
        by_sensor_percent = by_sensor * 0.0

    print("\n=== Effective weight by sensor (permutation importance, normalized) ===")
    for sensor, percent in by_sensor_percent.sort_values(ascending=False).items():
        print(f"  {sensor:30s} {percent:5.1f}%")

    print("\n=== Top individual features by importance ===")
    for feature, value in importances.sort_values(ascending=False).head(10).items():
        print(f"  {feature:30s} {value:.5f}")

    threshold = args.correlation_threshold
    print(
        "\n=== Near-duplicate features within a sensor group "
        f"(|corr| >= {threshold}) ==="
    )
    redundant = find_redundant_pairs(reference_df, feature_cols, threshold)
    if not redundant:
        print("  None found.")
    else:
        for sensor, first, second, value in sorted(
            redundant,
            key=lambda row: -row[3],
        ):
            print(f"  [{sensor}] {first} <-> {second}  (corr={value:.3f})")
        print(
            "\n  These pairs move almost identically across the reference set. If one "
            "sensor dominates the importance split, consider whether redundant features "
            "from that group should be pruned during a future model revision."
        )


if __name__ == "__main__":
    main()
