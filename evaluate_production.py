"""
Production evaluation for Project F.

This evaluator deliberately contains NO inference implementation of its own.
Every prediction is produced by predict_realtime.SpindleMonitor, exactly as in
deployment, and only the returned production outputs are compared with labels.

Usage:
    python evaluate_production.py

Output:
    results/production_evaluation.md
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

import config
from predict_realtime import SpindleMonitor


LABEL_COLUMN = "health_status"
REPORT_PATH = os.path.join(config.RESULTS_DIR, "evaluation_report.md")


@dataclass
class BinaryMetrics:
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    accuracy: float
    f1: float
    false_alarm_rate: float
    miss_rate: float


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den else float("nan")


def _fmt(value: float, digits: int = 4) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{float(value):.{digits}f}"


def _fmt_pct(value: float, digits: int = 2) -> str:
    if value is None or not math.isfinite(float(value)):
        return "N/A"
    return f"{100.0 * float(value):.{digits}f}%"


def _binary_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> BinaryMetrics:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return BinaryMetrics(
        tp=int(tp),
        fp=int(fp),
        tn=int(tn),
        fn=int(fn),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        false_alarm_rate=_safe_div(fp, fp + tn),
        miss_rate=_safe_div(fn, fn + tp),
    )


def _ranking_metrics(y_true: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    if np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    return float(roc_auc_score(y_true, score)), float(average_precision_score(y_true, score))


def _expected_calibration_error(y_true: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """Standard fixed-width expected calibration error over [0, 1]."""
    prob = np.clip(np.asarray(prob, dtype=float), 0.0, 1.0)
    y_true = np.asarray(y_true, dtype=int)
    edges = np.linspace(0.0, 1.0, bins + 1)
    # Put p=1.0 in the final bin rather than creating an out-of-range index.
    bucket = np.minimum(np.digitize(prob, edges[1:-1], right=False), bins - 1)

    total = len(prob)
    if total == 0:
        return float("nan")

    ece = 0.0
    for i in range(bins):
        mask = bucket == i
        if not np.any(mask):
            continue
        observed = float(np.mean(y_true[mask]))
        predicted = float(np.mean(prob[mask]))
        ece += (int(mask.sum()) / total) * abs(observed - predicted)
    return float(ece)


def _normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower()


def _validate_dataset(raw: pd.DataFrame) -> None:
    required = [config.COL_TIMESTAMP] + config.RAW_SENSOR_COLS + [LABEL_COLUMN]
    missing = [col for col in required if col not in raw.columns]
    if missing:
        raise ValueError(
            "Production evaluation requires timestamp, all production sensor columns, "
            f"and ground-truth '{LABEL_COLUMN}'. Missing: {missing}"
        )

    labels = set(_normalize_labels(raw[LABEL_COLUMN]).dropna().unique())
    allowed = {"normal", "warning", "critical"}
    unknown = sorted(labels - allowed)
    if unknown:
        raise ValueError(f"Unsupported {LABEL_COLUMN} value(s): {unknown}. Expected {sorted(allowed)}.")


def replay_production(raw: pd.DataFrame) -> pd.DataFrame:
    """Replay raw rows through the actual production SpindleMonitor only."""
    monitor = SpindleMonitor()
    predictions: list[dict] = []

    for row in raw.itertuples(index=False):
        reading = {
            config.COL_VIBRATION: float(getattr(row, config.COL_VIBRATION)),
            config.COL_TEMPERATURE: float(getattr(row, config.COL_TEMPERATURE)),
            config.COL_CURRENT: float(getattr(row, config.COL_CURRENT)),
        }
        result = monitor.update(reading)
        if result is None:
            continue

        failure_probability = result["failure_probability"]
        predictions.append({
            config.COL_TIMESTAMP: pd.Timestamp(getattr(row, config.COL_TIMESTAMP)),
            "anomaly_score": float(result["anomaly_score"]),
            "health_raw": float(result["health_raw"]),
            "health_state": float(result["health_state"]),
            "remaining_days": float(result["remaining_days"]),
            "maintenance_level": str(result["maintenance"]["level"]),
            "maintenance_reason": str(result["maintenance"]["reason"]),
            **{f"failure_probability_{h}d": float(p) for h, p in failure_probability.items()},
        })

    if not predictions:
        raise ValueError(
            "The production monitor emitted no predictions. The dataset may be shorter than the "
            "production warm-up requirements."
        )
    return pd.DataFrame(predictions)


def _next_critical_time(raw: pd.DataFrame) -> pd.Series:
    """For each raw row, return the next timestamp labeled critical, including the current row."""
    labels = _normalize_labels(raw[LABEL_COLUMN]).to_numpy()
    timestamps = pd.to_datetime(raw[config.COL_TIMESTAMP]).to_numpy(dtype="datetime64[ns]")
    out = np.full(len(raw), np.datetime64("NaT"), dtype="datetime64[ns]")
    next_critical = np.datetime64("NaT")
    for i in range(len(raw) - 1, -1, -1):
        if labels[i] == "critical":
            next_critical = timestamps[i]
        out[i] = next_critical
    return pd.Series(out, index=pd.to_datetime(raw[config.COL_TIMESTAMP]))


def evaluate(raw: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    labels = raw[[config.COL_TIMESTAMP, LABEL_COLUMN]].copy()
    labels[config.COL_TIMESTAMP] = pd.to_datetime(labels[config.COL_TIMESTAMP])
    labels[LABEL_COLUMN] = _normalize_labels(labels[LABEL_COLUMN])

    merged = predictions.merge(labels, on=config.COL_TIMESTAMP, how="left", validate="one_to_one")
    if merged[LABEL_COLUMN].isna().any():
        raise ValueError("Some production predictions could not be aligned to ground-truth timestamps.")

    # Health estimation is kept separate from maintenance policy. Ground truth is
    # binary degraded/not-degraded: warning or critical is positive. The production
    # health estimate is converted to the same binary view using the configured
    # inspection threshold; its continuous risk score is 100-health_state.
    y_health = (merged[LABEL_COLUMN] != "normal").astype(int).to_numpy()
    y_health_pred = (merged["health_state"] <= config.MAINTENANCE_HEALTH_INSPECT).astype(int).to_numpy()
    health_risk = np.clip((100.0 - merged["health_state"].to_numpy(dtype=float)) / 100.0, 0.0, 1.0)
    health_roc, health_pr = _ranking_metrics(y_health, health_risk)
    health_cls = _binary_metrics(y_health, y_health_pred)

    # A failure is represented by the available ground-truth critical state. For
    # probability/recommendation evaluation, positive means a critical state occurs
    # now or within the configured maintenance horizon.
    next_critical = _next_critical_time(raw)
    pred_times = pd.to_datetime(merged[config.COL_TIMESTAMP])
    critical_times = next_critical.reindex(pred_times).to_numpy(dtype="datetime64[ns]")
    current_times = pred_times.to_numpy(dtype="datetime64[ns]")
    horizon_delta = np.timedelta64(int(round(config.MAINTENANCE_HORIZON_DAYS * 24 * 60)), "m")
    has_critical = ~pd.isna(critical_times)
    y_failure = (has_critical & (critical_times >= current_times) & (critical_times <= current_times + horizon_delta)).astype(int)

    horizon = config.MAINTENANCE_HORIZON_DAYS
    probability_column = f"failure_probability_{horizon}d"
    if probability_column not in merged.columns:
        raise ValueError(
            f"Production output does not include configured maintenance horizon {horizon}d. "
            f"Available probability columns: {[c for c in merged if c.startswith('failure_probability_')]}"
        )
    failure_prob = np.clip(merged[probability_column].to_numpy(dtype=float), 0.0, 1.0)
    failure_roc, failure_pr = _ranking_metrics(y_failure, failure_prob)
    failure_brier = float(brier_score_loss(y_failure, failure_prob))
    failure_ece = _expected_calibration_error(y_failure, failure_prob)

    y_maintenance_pred = (merged["maintenance_level"] != "OK").astype(int).to_numpy()
    maintenance_metrics = _binary_metrics(y_failure, y_maintenance_pred)

    return {
        "merged": merged,
        "health_y": y_health,
        "health_roc": health_roc,
        "health_pr": health_pr,
        "health_metrics": health_cls,
        "failure_y": y_failure,
        "failure_roc": failure_roc,
        "failure_pr": failure_pr,
        "failure_brier": failure_brier,
        "failure_ece": failure_ece,
        "maintenance_metrics": maintenance_metrics,
        "probability_column": probability_column,
    }


def write_report(raw: pd.DataFrame, predictions: pd.DataFrame, metrics: dict) -> None:
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    hm = metrics["health_metrics"]
    mm = metrics["maintenance_metrics"]
    emitted = len(predictions)
    skipped = len(raw) - emitted
    health_bad = int(metrics["health_y"].sum())
    failure_positive = int(metrics["failure_y"].sum())

    lines = [
        "# Production Evaluation Report",
        "",
        "## Dataset",
        "",
        f"- Source: `{os.path.relpath(config.PREDICT_DATA_PATH, config.ROOT_DIR)}`",
        f"- Raw samples: {len(raw):,}",
        f"- Production predictions evaluated: {emitted:,}",
        f"- Warm-up rows with no production output: {skipped:,}",
        f"- Labeled degraded rows among evaluated predictions (warning/critical): {health_bad:,}",
        f"- Rows with a ground-truth critical state now/within {config.MAINTENANCE_HORIZON_DAYS:g} day(s): {failure_positive:,}",
        "",
        "Every evaluated prediction above was returned directly by `predict_realtime.SpindleMonitor.update()`.",
        "The evaluator contains no duplicate anomaly, Kalman, trend, probability, RUL, or maintenance inference path.",
        "",
        "## Health Estimation",
        "",
        f"Binary ground truth: `health_status != normal`. Predicted degraded: `health_state <= {config.MAINTENANCE_HEALTH_INSPECT}`.",
        f"Continuous risk score for ROC/PR: `(100 - health_state) / 100`.",
        "",
        f"- ROC-AUC: {_fmt(metrics['health_roc'])}",
        f"- PR-AUC: {_fmt(metrics['health_pr'])}",
        f"- Accuracy: {_fmt_pct(hm.accuracy)}",
        f"- Precision: {_fmt_pct(hm.precision)}",
        f"- Recall: {_fmt_pct(hm.recall)}",
        f"- F1-score: {_fmt_pct(hm.f1)}",
        "",
        "Confusion Matrix:",
        "",
        "| Actual \\ Predicted | Normal | Degraded |",
        "|---|---:|---:|",
        f"| Normal | {hm.tn:,} | {hm.fp:,} |",
        f"| Degraded | {hm.fn:,} | {hm.tp:,} |",
        "",
        "## Failure Probability",
        "",
        f"Ground truth: a `critical` label occurs now or within the configured {config.MAINTENANCE_HORIZON_DAYS:g}-day maintenance horizon.",
        f"Production probability field: `{metrics['probability_column']}`.",
        "",
        f"- ROC-AUC: {_fmt(metrics['failure_roc'])}",
        f"- PR-AUC: {_fmt(metrics['failure_pr'])}",
        f"- Brier Score: {_fmt(metrics['failure_brier'])}",
        f"- Calibration Error (10-bin ECE): {_fmt(metrics['failure_ece'])}",
        "",
        "## Remaining Useful Life",
        "",
        "SKIPPED — the configured evaluation dataset does not contain an independent true-RUL label/column. "
        "The production `remaining_days` output is collected, but MAE/RMSE are not fabricated from health-status labels.",
        "",
        "## Maintenance Recommendation",
        "",
        f"Predicted positive: production maintenance level is `WARN` or `CRITICAL`. Actual positive: a ground-truth `critical` state occurs now or within {config.MAINTENANCE_HORIZON_DAYS:g} day(s).",
        "",
        f"- True Positives: {mm.tp:,}",
        f"- False Positives: {mm.fp:,}",
        f"- True Negatives: {mm.tn:,}",
        f"- False Negatives: {mm.fn:,}",
        f"- Precision: {_fmt_pct(mm.precision)}",
        f"- Recall: {_fmt_pct(mm.recall)}",
        f"- False Alarm Rate: {_fmt_pct(mm.false_alarm_rate)}",
        f"- Miss Rate: {_fmt_pct(mm.miss_rate)}",
        "",
        "## Overall Result",
        "",
        "EVALUATION COMPLETED",
        "",
        "No model-quality PASS/FAIL thresholds are defined by the production-evaluation architecture, so this report does not invent an acceptance gate.",
        "",
        "The production pipeline was evaluated directly using `predict_realtime.SpindleMonitor`. All reported metrics are computed from outputs returned by the same inference implementation used for deployment.",
        "",
    ]

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main() -> None:
    raw = pd.read_csv(config.PREDICT_DATA_PATH)
    _validate_dataset(raw)
    raw[config.COL_TIMESTAMP] = pd.to_datetime(raw[config.COL_TIMESTAMP])
    raw = raw.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)

    predictions = replay_production(raw)
    metrics = evaluate(raw, predictions)
    write_report(raw, predictions, metrics)

    print("Production evaluation complete.")
    print(f"Predictions evaluated: {len(predictions):,} / {len(raw):,}")
    print(f"Report: {REPORT_PATH}")


if __name__ == "__main__":
    main()
