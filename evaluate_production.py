"""
Production evaluation for Project F.

This evaluator deliberately contains NO inference implementation of its own.
Every prediction is produced by predict_realtime.SpindleMonitor, exactly as in
deployment, and only the returned production outputs are compared with labels.

Usage:
    python evaluate_production.py

Output:
    results/evaluation_report.md
    results/evaluation_report.png
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import hashlib

import config
from predict_realtime import SpindleMonitor
import predict_realtime


LABEL_COLUMN = "health_status"
REPORT_PATH = os.path.join(config.RESULTS_DIR, "evaluation_report.md")
PNG_REPORT_PATH = os.path.join(config.RESULTS_DIR, "evaluation_report.png")


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
    total_rows = len(raw)
    processed_rows = 0

    for row in raw.itertuples(index=False):
        processed_rows += 1
        reading = {col: float(getattr(row, col)) for col in config.RAW_SENSOR_COLS}
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

        progress = processed_rows / total_rows if total_rows else 1.0
        filled = int(progress * 30)
        bar = "■" * filled + "-" * (30 - filled)
        percent = int(progress * 100)
        print(f"Processing: [{bar}] {percent}% ({processed_rows}/{total_rows})", end="\r", flush=True)

    print()

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
        f"- True Positives (TP): {mm.tp:,}",
        f"- True Negatives (TN): {mm.tn:,}",
        f"- False Positives (FP): {mm.fp:,}",
        f"- False Negatives (FN): {mm.fn:,}",
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


FIG_W = 8.5  # letter width, portrait — height grows to fit content (see below)
MARGIN_IN = 0.5
CONTENT_W_IN = FIG_W - 2 * MARGIN_IN

# Type: a clean geometric sans for headings/emphasis, a highly-readable sans
# for body/data, and a monospace for technical tokens (report id, timestamps,
# hashes). Preference lists, not hardcoded names — the machine that runs this
# script may not have the same fonts installed as the machine that wrote it,
# and matplotlib silently substitutes DejaVu Sans (with console warnings) for
# any missing family. We pick the first available match per role instead, so
# there's no warning spam and the layout math (which depends on actually
# measuring these exact fonts) matches what gets drawn.
def _available_font_names():
    from matplotlib.font_manager import fontManager
    return {f.name for f in fontManager.ttflist}


def _pick_font(preferred, fallback):
    names = _available_font_names()
    for name in preferred:
        if name in names:
            return name
    return fallback


FONT_HEAD = _pick_font(
    ["Poppins", "Montserrat", "Calibri", "Segoe UI", "Helvetica Neue", "Arial",
     "Liberation Sans", "DejaVu Sans"], "sans-serif")
FONT_BODY = _pick_font(
    ["Carlito", "Calibri", "Segoe UI", "Helvetica Neue", "Arial", "Liberation Sans",
     "DejaVu Sans"], "sans-serif")
FONT_MONO = _pick_font(
    ["Liberation Mono", "Consolas", "Menlo", "Courier New", "DejaVu Sans Mono"],
    "monospace")

# Only "normal"/"bold" — not numeric weights like 600. A numeric weight with
# no matching font file (common on fallback fonts, which usually only ship
# regular + bold) triggers the same kind of silent substitution + warning.

INK = "#161f36"
SUBINK = "#5a6579"
MUTED = "#8992a3"
RULE = "#e3e7ee"
ACCENT = "#2f6fb0"
ACCENT_SOFT = "#eaf1fa"
CARD_BG = "#f6f8fb"
CARD_BORDER = "#e1e6ee"
BANNER_BG = "#f4f7fb"


def _kwf(weight="normal", family=FONT_BODY):
    return {"fontfamily": family, "fontweight": weight}


# ---------------------------------------------------------------------------
# Text measurement. matplotlib's built-in ax.text(..., wrap=True) is not
# reliable — it can silently fail to wrap (text runs off the page) depending
# on backend/draw order. Instead we measure real rendered glyph width/height
# via a throwaway renderer and wrap ourselves, so line breaks and block
# heights are computed from what will actually render, not guessed.
# ---------------------------------------------------------------------------
_measure_fig = None
_measure_renderer = None


def _renderer():
    global _measure_fig, _measure_renderer
    if _measure_renderer is None:
        _measure_fig = plt.figure(figsize=(1, 1), dpi=100)
        _measure_renderer = _measure_fig.canvas.get_renderer()
    return _measure_renderer, _measure_fig.dpi


def _text_size_in(text, fontsize, weight="normal", family=FONT_BODY):
    """Rendered (width, height) in inches for one line of text. Independent
    of final page size — text physical size only depends on fontsize/weight."""
    from matplotlib.font_manager import FontProperties, findfont
    renderer, dpi = _renderer()
    fp = FontProperties(family=family, size=fontsize, weight=weight)
    w, h, d = renderer.get_text_width_height_descent(text, fp, False)
    return w / dpi, h / dpi


def _wrap_lines(text, fontsize, max_width_in, weight="normal", family=FONT_BODY):
    """Greedy word-wrap using measured widths — guaranteed to fit
    max_width_in, unlike matplotlib's wrap=True."""
    words = text.split()
    lines, cur = [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        tw, _ = _text_size_in(trial, fontsize, weight, family)
        if tw <= max_width_in or not cur:
            cur = trial
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


class _Layout:
    """Top-down cursor in real inches from the top of the page. Two passes:
    pass 1 (draw=False) just measures/accumulates total height needed; pass 2
    (draw=True, against a figure already sized to fit) does the actual
    drawing. Same code, same measurements, both times — so what gets drawn
    always matches what was budgeted for, eliminating the overlap/overflow
    bugs a hand-guessed figure-fraction layout was prone to."""

    LEFT_IN = MARGIN_IN
    RIGHT_IN = FIG_W - MARGIN_IN

    def __init__(self, fig=None, fig_h=None):
        self.fig = fig
        self.fig_h = fig_h
        self.draw = fig is not None
        self.y_in = MARGIN_IN

    def _fx(self, x_in):
        return x_in / FIG_W

    def _fy(self, y_in):
        return 1.0 - (y_in / self.fig_h)

    def gap(self, h_in):
        self.y_in += h_in

    def line(self, s, fontsize, x_in=None, ha="left", family=FONT_BODY, weight="normal",
              **kw):
        """Single line, no wrapping (use for short, bounded-length strings)."""
        _, h = _text_size_in(s, fontsize, weight, family)
        if self.draw:
            x = self.RIGHT_IN if ha == "right" else (x_in if x_in is not None else self.LEFT_IN)
            self.fig.text(self._fx(x), self._fy(self.y_in), s, fontsize=fontsize,
                          va="top", ha=ha, fontfamily=family, fontweight=weight, **kw)
        self.gap(h * 1.35)

    def title(self, s, accent=True):
        if self.draw and accent:
            bar_w_in, bar_h_in = 0.05, 0.17
            y0 = self._fy(self.y_in + bar_h_in * 1.05)
            self.fig.add_artist(plt.Rectangle(
                (self._fx(self.LEFT_IN), y0), bar_w_in / FIG_W, bar_h_in / self.fig_h,
                transform=self.fig.transFigure, facecolor=ACCENT, edgecolor="none"))
        self.line(s, 13.5, x_in=self.LEFT_IN + (0.14 if accent else 0), family=FONT_HEAD,
                  weight="bold", color=INK)
        self.gap(0.03)

    def wrapped(self, text, fontsize=8.0, weight="normal", family=FONT_BODY, color=SUBINK,
                gap_after=0.05):
        lines = _wrap_lines(text, fontsize, CONTENT_W_IN, weight, family)
        _, line_h = _text_size_in("Ag", fontsize, weight, family)
        line_h *= 1.38
        if self.draw:
            for i, ln in enumerate(lines):
                self.fig.text(self._fx(self.LEFT_IN), self._fy(self.y_in + i * line_h), ln,
                              fontsize=fontsize, va="top", ha="left", color=color,
                              fontfamily=family, fontweight=weight)
        self.gap(line_h * len(lines))
        self.gap(gap_after)

    def config_banner(self, title, items, fontsize_label=6.3, fontsize_value=10.0,
                       pad_in=0.14, cell_h_in=0.52):
        """Soft panel with a small caption plus a row of label/value cells —
        one per config constant. Kept structurally identical to the metric
        cards elsewhere on the page (small muted label, bold value) so the
        exact policy this run used is scannable at a glance instead of
        buried in a run-on sentence of CONSTANT_NAME=value pairs.

        Background, title, and cells are all drawn in ONE axes (like
        _metric_cards does) rather than a figure-level patch behind a
        separate axes — matplotlib draws figure-level artists (fig.add_artist)
        above ALL axes regardless of call order, so a fig-level background
        would silently paint over a separate cells axes. Single axes avoids
        that entirely: normal draw order within one axes is call order."""
        _, title_h = _text_size_in("Ag", 8.4, "bold", FONT_HEAD)
        title_h *= 1.35
        block_h = pad_in + title_h + 0.06 + cell_h_in + pad_in
        if self.draw:
            rect = self._rect(self.LEFT_IN, self.y_in, CONTENT_W_IN, block_h)
            ax = self.fig.add_axes(rect)
            ax.axis("off")
            ax.set_xlim(0, CONTENT_W_IN)
            ax.set_ylim(0, block_h)
            ax.add_patch(FancyBboxPatch(
                (0, 0), CONTENT_W_IN, block_h, boxstyle="round,pad=0,rounding_size=0.06",
                facecolor=BANNER_BG, edgecolor=RULE, linewidth=0.8, transform=ax.transData))
            ax.text(pad_in, block_h - pad_in, title, fontsize=8.4, va="top", ha="left",
                    color=SUBINK, fontfamily=FONT_HEAD, fontweight="bold")

            cells_top = block_h - (pad_in + title_h + 0.06)
            n = len(items)
            gap_in = 0.12
            usable_w = CONTENT_W_IN - 2 * pad_in
            cw_in = (usable_w - gap_in * (n - 1)) / n
            cell_mid = cells_top - cell_h_in * 0.5
            for i, (label, value) in enumerate(items):
                x0 = pad_in + i * (cw_in + gap_in)
                x_center = x0 + cw_in / 2
                if i > 0:
                    sep_x = x0 - gap_in / 2
                    ax.add_line(plt.Line2D([sep_x, sep_x],
                                            [cells_top - cell_h_in * 0.85,
                                             cells_top - cell_h_in * 0.15],
                                            color=RULE, linewidth=0.8))
                ax.text(x_center, cell_mid + cell_h_in * 0.16, value, fontsize=fontsize_value,
                        ha="center", va="center", fontfamily=FONT_HEAD, fontweight="bold",
                        color=ACCENT)
                # Shrink the label to fit its cell — different fonts (whichever
                # was actually available on this machine) measure different
                # widths for the same string, so a fixed fontsize that fit one
                # font can overflow into the next cell under another.
                label_fs = fontsize_label
                avail_w = cw_in - 0.03
                lw, _ = _text_size_in(label, label_fs, "bold", FONT_BODY)
                while lw > avail_w and label_fs > 5.0:
                    label_fs -= 0.3
                    lw, _ = _text_size_in(label, label_fs, "bold", FONT_BODY)
                ax.text(x_center, cell_mid - cell_h_in * 0.24, label, fontsize=label_fs,
                        ha="center", va="center", fontfamily=FONT_BODY, fontweight="bold",
                        color=MUTED)
        self.y_in += block_h
        self.gap(0.16)

    def rule(self, color=RULE, lw=1.0, pre_gap=0.05, post_gap=0.15):
        self.gap(pre_gap)
        if self.draw:
            y = self._fy(self.y_in)
            self.fig.add_artist(plt.Line2D([self._fx(self.LEFT_IN), self._fx(self.RIGHT_IN)],
                                            [y, y], transform=self.fig.transFigure,
                                            color=color, linewidth=lw))
        self.gap(post_gap)

    def kv_table(self, rows, row_h_in=0.21):
        h_in = row_h_in * len(rows)
        if self.draw:
            rect = self._rect(self.LEFT_IN, self.y_in, CONTENT_W_IN, h_in)
            _kv_table(self.fig, rect, rows)
        self.y_in += h_in
        self.gap(0.10)

    def cards(self, items, h_in=0.58):
        if self.draw:
            rect = self._rect(self.LEFT_IN, self.y_in, CONTENT_W_IN, h_in)
            _metric_cards(self.fig, rect, items)
        self.y_in += h_in
        self.gap(0.16)

    def confusion_and_kv(self, m, row_pos, col_pos, kv_rows, h_in=1.35, kv_row_h=0.19):
        """Confusion-matrix heatmap (left) + a key/value table (right). The
        heatmap's own title sits ABOVE its axes box in matplotlib (drawn via
        padding, not inside the given rect) — we reserve dedicated vertical
        budget for it up front so it can never bleed into whatever was drawn
        above this block, which is exactly the overlap seen before."""
        title_reserve_in = 0.32
        cm_w_in = 3.1
        top_in = self.y_in + title_reserve_in
        if self.draw:
            cm_rect = self._rect(self.LEFT_IN + 0.15, top_in, cm_w_in, h_in)
            ax = self.fig.add_axes(cm_rect)
            _confusion_heatmap(ax, m, row_pos, col_pos, "Confusion Matrix")
            kv_h_in = kv_row_h * len(kv_rows)
            kv_top_in = top_in + (h_in - kv_h_in) / 2.0
            kv_rect = self._rect(self.LEFT_IN + cm_w_in + 0.55, kv_top_in,
                                  self.RIGHT_IN - (self.LEFT_IN + cm_w_in + 0.55), kv_h_in)
            _kv_table(self.fig, kv_rect, kv_rows)
        self.y_in = top_in + h_in
        self.gap(0.16)

    def _rect(self, x_in, y_top_in, w_in, h_in):
        """[left, bottom, width, height] figure-fraction rect for add_axes,
        from an inch-space top-left box."""
        return [self._fx(x_in), self._fy(y_top_in + h_in), w_in / FIG_W, h_in / self.fig_h]


def _kv_table(fig, rect, rows):
    """rows: list of (label, value). Drawn as a borderless two-column table
    inside the figure-fraction rectangle [left, bottom, width, height]."""
    ax = fig.add_axes(rect)
    ax.axis("off")
    n = len(rows)
    row_h = 1.0 / n
    for i, (label, value) in enumerate(rows):
        y = 1.0 - (i + 0.5) * row_h
        ax.text(0.0, y, label, fontsize=8.7, color=SUBINK, ha="left", va="center",
                fontfamily=FONT_BODY)
        ax.text(1.0, y, value, fontsize=8.7, color=INK, ha="right", va="center",
                fontfamily=FONT_HEAD, fontweight="bold")
        if i < n - 1:
            ax.add_line(plt.Line2D([0, 1], [y - row_h / 2, y - row_h / 2],
                                    color=RULE, linewidth=0.7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)


def _metric_cards(fig, rect, items, ncols=None):
    """items: list of (label, value). Drawn as evenly spaced rounded cards
    inside the figure-fraction rectangle [left, bottom, width, height]."""
    n = len(items)
    ncols = ncols or n
    ax = fig.add_axes(rect)
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    gap = 0.025
    cw = (1.0 - gap * (ncols - 1)) / ncols
    for i, (label, value) in enumerate(items):
        x0 = i * (cw + gap)
        ax.add_patch(FancyBboxPatch((x0, 0.03), cw, 0.94,
                                     boxstyle="round,pad=0,rounding_size=0.05",
                                     facecolor=CARD_BG, edgecolor=CARD_BORDER, linewidth=0.9,
                                     transform=ax.transData))
        ax.text(x0 + cw / 2, 0.63, value, ha="center", va="center", fontsize=14.5,
                fontfamily=FONT_HEAD, fontweight="bold", color=ACCENT)
        ax.text(x0 + cw / 2, 0.23, label, ha="center", va="center", fontsize=7.4,
                fontfamily=FONT_BODY, color=SUBINK)


def _confusion_heatmap(ax, m: BinaryMetrics, row_pos: str, col_pos: str, title: str) -> None:
    """row_pos/col_pos: short label for the *positive* class on each axis
    (e.g. 'Degraded', 'Flagged') — kept short so the two column/row labels
    don't collide in a narrow 2x2 grid."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("report_blue", ["#eef3fa", ACCENT])
    mat = np.array([[m.tn, m.fp], [m.fn, m.tp]])
    vmax = mat.max() if mat.max() else 1
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=vmax)
    for i in range(2):
        for j in range(2):
            val = mat[i, j]
            color = "white" if val > vmax * 0.55 else INK
            ax.text(j, i, f"{val:,}", ha="center", va="center", color=color, fontsize=10.5,
                    fontfamily=FONT_HEAD, fontweight="bold")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", col_pos], fontsize=7.4, fontfamily=FONT_BODY)
    ax.set_yticklabels(["Normal", row_pos], fontsize=7.4, fontfamily=FONT_BODY)
    ax.set_title(title, fontsize=9.8, fontfamily=FONT_HEAD, fontweight="bold", color=INK, pad=7)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def _page_border(fig, fig_h):
    m = MARGIN_IN * 0.55
    fig.add_artist(FancyBboxPatch(
        (m / FIG_W, m / fig_h), 1 - 2 * m / FIG_W, 1 - 2 * m / fig_h,
        boxstyle="round,pad=0,rounding_size=0.006", transform=fig.transFigure,
        fill=False, edgecolor=RULE, linewidth=1.3))


def _report_id(raw: pd.DataFrame, metrics: dict, generated_at: pd.Timestamp) -> str:
    """Short deterministic fingerprint of (dataset, config, results, generation
    time) — lets two people confirm they're looking at the same run without
    re-computing all the metrics themselves."""
    mm = metrics["maintenance_metrics"]
    fingerprint = "|".join([
        str(len(raw)),
        str(predict_realtime.metadata.get("pipeline_hash", "")),
        f"{config.MAINTENANCE_HEALTH_INSPECT}-{config.FAILURE_HEALTH_THRESHOLD}",
        f"{config.MAINTENANCE_PROB_URGENT}-{config.MAINTENANCE_PROB_PLAN}",
        f"{mm.tp}-{mm.fp}-{mm.tn}-{mm.fn}",
        generated_at.isoformat(),
    ])
    return hashlib.sha256(fingerprint.encode()).hexdigest()[:10].upper()


def _build_dashboard(L, raw, predictions, metrics, report_id, generated_at,
                      model_trained_at, pipeline_hash):
    """Draws (or, if L.draw is False, just measures) the full page. Called
    twice by write_dashboard_png: once to find the required page height,
    once for real against a figure sized to fit."""
    hm = metrics["health_metrics"]
    mm = metrics["maintenance_metrics"]
    emitted = len(predictions)
    skipped = len(raw) - emitted
    health_bad = int(metrics["health_y"].sum())
    failure_positive = int(metrics["failure_y"].sum())

    # ---- Header ----
    L.line("Production Evaluation Report", 19, family=FONT_HEAD, weight="bold", color=INK)
    if L.draw:
        L.fig.text(L._fx(L.RIGHT_IN), L._fy(MARGIN_IN + 0.04), f"REPORT ID  {report_id}",
                   fontsize=8, color=SUBINK, ha="right", va="top", fontfamily=FONT_MONO)
    L.gap(0.11)
    L.wrapped(f"{os.path.relpath(config.PREDICT_DATA_PATH, config.ROOT_DIR)}  ·  "
              f"{emitted:,} predictions evaluated  ·  predict_realtime.SpindleMonitor "
              f"(production inference path)", 8.6, family=FONT_BODY, color=SUBINK,
              gap_after=0.04)
    L.wrapped(f"Generated {generated_at.strftime('%Y-%m-%d %H:%M:%S')}  ·  "
              f"Model trained {model_trained_at}  ·  pipeline {pipeline_hash[:10]}",
              7.5, family=FONT_MONO, color=MUTED, gap_after=0.02)
    L.rule(lw=1.5, color=INK)

    # ---- Report configuration ----
    L.config_banner("Configuration used for this report", [
        ("FAILURE_HEALTH_THRESHOLD", f"{config.FAILURE_HEALTH_THRESHOLD}"),
        ("MAINTENANCE_HEALTH_INSPECT", f"{config.MAINTENANCE_HEALTH_INSPECT}"),
        ("MAINTENANCE_PROB_URGENT", f"{config.MAINTENANCE_PROB_URGENT}"),
        ("MAINTENANCE_PROB_PLAN", f"{config.MAINTENANCE_PROB_PLAN}"),
        ("MAINTENANCE_HORIZON_DAYS", f"{config.MAINTENANCE_HORIZON_DAYS:g}"),
    ])

    # ---- Dataset ----
    L.title("Dataset")
    L.kv_table([
        ("Raw samples", f"{len(raw):,}"),
        ("Predictions evaluated", f"{emitted:,}"),
        ("Warm-up rows with no output", f"{skipped:,}"),
        ("Labeled degraded rows (warning/critical)", f"{health_bad:,}"),
        (f"Rows with critical state within {config.MAINTENANCE_HORIZON_DAYS:g} day(s)",
         f"{failure_positive:,}"),
    ])
    L.wrapped("Every prediction below was returned directly by SpindleMonitor.update() — "
              "no duplicate anomaly, Kalman, trend, probability, RUL, or maintenance logic.")
    L.rule()

    # ---- Health Estimation ----
    L.title("Health Estimation")
    L.wrapped(f"Ground truth: health_status != normal.  Predicted degraded: health_state <= "
              f"{config.MAINTENANCE_HEALTH_INSPECT}.  Risk score: (100 - health_state) / 100.")
    L.cards([
        ("ROC-AUC", _fmt(metrics["health_roc"])),
        ("PR-AUC", _fmt(metrics["health_pr"])),
        ("Accuracy", _fmt_pct(hm.accuracy)),
        ("F1-score", _fmt_pct(hm.f1)),
    ])
    L.confusion_and_kv(hm, "Degraded", "Degraded", [
        ("Precision", _fmt_pct(hm.precision)),
        ("Recall", _fmt_pct(hm.recall)),
        ("F1-score", _fmt_pct(hm.f1)),
        ("True Neg. / False Pos.", f"{hm.tn:,} / {hm.fp:,}"),
        ("False Neg. / True Pos.", f"{hm.fn:,} / {hm.tp:,}"),
    ])
    L.rule()

    # ---- Failure Probability ----
    L.title("Failure Probability")
    L.wrapped(f"Ground truth: a critical label occurs now or within the configured "
              f"{config.MAINTENANCE_HORIZON_DAYS:g}-day horizon.  Field: "
              f"{metrics['probability_column']}.")
    L.cards([
        ("ROC-AUC", _fmt(metrics["failure_roc"])),
        ("PR-AUC", _fmt(metrics["failure_pr"])),
        ("Brier Score", _fmt(metrics["failure_brier"])),
        ("Calib. Error (ECE)", _fmt(metrics["failure_ece"])),
    ])
    L.rule()

    # ---- Remaining Useful Life ----
    L.title("Remaining Useful Life")
    L.wrapped("SKIPPED — no independent true-RUL label/column in this dataset. "
              "remaining_days is collected, but MAE/RMSE are not fabricated from "
              "health-status labels.")
    L.rule()

    # ---- Maintenance Recommendation ----
    L.title("Maintenance Recommendation")
    L.wrapped(f"Predicted positive: level is WARN or CRITICAL.  Actual positive: a "
              f"ground-truth critical state now or within "
              f"{config.MAINTENANCE_HORIZON_DAYS:g} day(s).")
    L.confusion_and_kv(mm, "At risk", "Flagged", [
        ("Precision", _fmt_pct(mm.precision)),
        ("Recall", _fmt_pct(mm.recall)),
        ("False Alarm Rate", _fmt_pct(mm.false_alarm_rate)),
        ("Miss Rate", _fmt_pct(mm.miss_rate)),
        ("True Pos. / False Pos.", f"{mm.tp:,} / {mm.fp:,}"),
        ("True Neg. / False Neg.", f"{mm.tn:,} / {mm.fn:,}"),
    ])
    L.rule()

    # ---- Overall result / signature footer ----
    L.line("EVALUATION COMPLETED", 10.5, family=FONT_HEAD, weight="bold", color=ACCENT)
    if L.draw:
        L.fig.text(L._fx(L.RIGHT_IN), L._fy(L.y_in - 0.20), "Page 1 of 1", fontsize=7.6,
                   color=MUTED, ha="right", va="top", fontfamily=FONT_BODY)
    L.gap(0.05)
    L.wrapped("No PASS/FAIL threshold is defined by this evaluator, so none is invented. "
              "All metrics above come from SpindleMonitor's production output for the "
              "exact run identified by the Report ID.", gap_after=0.0)

    return L.y_in  # total content height consumed, in inches from the top


def write_dashboard_png(raw: pd.DataFrame, predictions: pd.DataFrame, metrics: dict) -> None:
    """Single portrait page, self-contained enough to stand as the official
    record of one evaluation run: every figure that appears in the markdown
    report also appears here, plus the exact policy configuration, model
    provenance, and a fingerprint identifying this specific run. Page height
    is computed from actual measured content (not guessed) so nothing is
    ever clipped or overlapping — width stays at letter (8.5in); height
    grows past 11in if the content needs it."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    generated_at = pd.Timestamp.now()
    report_id = _report_id(raw, metrics, generated_at)
    model_trained_at = predict_realtime.metadata.get("trained_at", "unknown")
    pipeline_hash = predict_realtime.metadata.get("pipeline_hash", "unknown")

    # Pass 1: measure only (no figure needed) to find the required height.
    dry = _Layout(fig=None)
    content_h_in = _build_dashboard(dry, raw, predictions, metrics, report_id, generated_at,
                                     model_trained_at, pipeline_hash)
    fig_h = max(11.0, content_h_in + MARGIN_IN)

    # Pass 2: draw for real against a figure sized to fit exactly.
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor="white")
    _page_border(fig, fig_h)
    real = _Layout(fig=fig, fig_h=fig_h)
    _build_dashboard(real, raw, predictions, metrics, report_id, generated_at,
                     model_trained_at, pipeline_hash)

    fig.savefig(PNG_REPORT_PATH, dpi=220, facecolor="white")
    plt.close(fig)


def write_dashboard_png(raw: pd.DataFrame, predictions: pd.DataFrame, metrics: dict) -> None:
    """Single portrait page, self-contained enough to stand as the official
    record of one evaluation run: every figure that appears in the markdown
    report also appears here, plus the exact policy configuration, model
    provenance, and a fingerprint identifying this specific run. Page height
    is computed from actual measured content (not guessed) so nothing is
    ever clipped or overlapping — width stays at letter (8.5in); height
    grows past 11in if the content needs it."""
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    generated_at = pd.Timestamp.now()
    report_id = _report_id(raw, metrics, generated_at)
    model_trained_at = predict_realtime.metadata.get("trained_at", "unknown")
    pipeline_hash = predict_realtime.metadata.get("pipeline_hash", "unknown")

    # Pass 1: measure only (no figure needed) to find the required height.
    dry = _Layout(fig=None)
    content_h_in = _build_dashboard(dry, raw, predictions, metrics, report_id, generated_at,
                                     model_trained_at, pipeline_hash)
    fig_h = max(11.0, content_h_in + MARGIN_IN)

    # Pass 2: draw for real against a figure sized to fit exactly.
    fig = plt.figure(figsize=(FIG_W, fig_h), facecolor="white")
    _page_border(fig, fig_h)
    real = _Layout(fig=fig, fig_h=fig_h)
    _build_dashboard(real, raw, predictions, metrics, report_id, generated_at,
                     model_trained_at, pipeline_hash)

    fig.savefig(PNG_REPORT_PATH, dpi=220, facecolor="white")
    plt.close(fig)


def main() -> None:
    raw = pd.read_csv(config.PREDICT_DATA_PATH)
    _validate_dataset(raw)
    raw[config.COL_TIMESTAMP] = pd.to_datetime(raw[config.COL_TIMESTAMP])
    raw = raw.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)

    predictions = replay_production(raw)
    metrics = evaluate(raw, predictions)
    write_report(raw, predictions, metrics)
    write_dashboard_png(raw, predictions, metrics)

    print("Production evaluation complete.")
    print(f"Predictions evaluated: {len(predictions):,} / {len(raw):,}")
    print(f"Report (markdown): {REPORT_PATH}")
    print(f"Report (PNG dashboard): {PNG_REPORT_PATH}")


if __name__ == "__main__":
    main()
