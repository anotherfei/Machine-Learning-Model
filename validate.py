"""
External reliability & trustworthiness test suite — NOT part of the
pipeline, and deliberately kept separate from it (formerly validate.py;
renamed because its scope grew past "accuracy validation" into a full
reliability/stability/consistency audit — see below).

This is the ONLY file in this repo that reads health_status. Nothing in
config.py, preprocessing.py, feature_engineering.py, isolation_forest.py,
kalman.py, trend_forecast.py, failure_probability.py, maintenance.py, or
predict_realtime.py ever loads it — that's the whole point of the
unsupervised design. This script exists purely so you can sanity-check
the trained model against health_status yourself, the same way you'd
manually compare against any other label you happen to have — it never
feeds back into training or inference. It also imports and directly
exercises predict_realtime.SpindleMonitor for one check (batch-vs-
real-time equivalence) — that's a read-only import for testing, not a
pipeline dependency.

Run this after every retrain:

    python reliability_suite.py

Four dimensions are tested, twelve checks total. Each check prints its
own numbers and a plain-language explanation of what they mean — there
is no PASS/WARN/FAIL label and no overall verdict, by design (see below
for why). The same findings are also rendered to validation_report.png.

  ACCURACY / DISCRIMINATION — can the model tell bad from normal at all,
  and how sure can we be given this is one trajectory with one event?
    1. Overfitting check (train/holdout split of the ACTUAL reference set)
    2. ROC-AUC / PR-AUC with block-bootstrap confidence intervals
    3. Threshold sweep + confusion matrices (raw threshold vs full
       maintenance output)

  CALIBRATION / RELIABILITY — when it gives a probability, is it right?
    4. Brier score per forecast horizon, with block-bootstrap CIs
    5. Reliability diagrams (predicted vs. observed frequency, binned)

  STABILITY — does the output stay sane under noise, degenerate inputs,
  and small changes to how the model was trained?
    6. Perturbation robustness (feature-level noise injection)
    7. Fault injection (extreme/degenerate/NaN inputs)
    8. Reference-window sensitivity (refit on nearby burn-in windows)

  CONSISTENCY — does it reproduce, and do independent code paths agree?
    9. Training reproducibility (two fits, identical data -> identical
       scores?)
   10. Batch vs. real-time equivalence (this script's vectorized
       backtest vs. predict_realtime.SpindleMonitor running tick-by-tick,
       on the same data)
   11. Cross-run regression ("golden snapshot" hash of a fixed output
       slice, compared against the previous run of this script)
   12. Threshold sanity check (health% thresholds vs. empirical sensor
       onset) — informational; see its own docstring for why this one
       is weaker evidence than the others and shouldn't be read the
       same way.

WHY NO VERDICT: an earlier version of this file rolled every check into
a PASS/WARN/FAIL and one bolded "OVERALL VERDICT." That was dropped on
purpose. Every threshold behind a verdict (a train/holdout gap in std
units, a CI touching 0.25, a correlation cutoff) is a reasonable
engineering default, not a certified statistical cutoff — so the
verdict was exactly as uncertain as the numbers underneath it, just
displayed with more apparent authority than those numbers earned. Twelve
checks covering different failure modes on one 6.9-day trajectory with
one labeled event don't collapse into one honest number. Read the
findings; there's no summary above them to skim to instead.
"""

import os
import io
import sys
import json
import hashlib
import warnings
import contextlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)

import config
import artifact_utils
import preprocessing
import feature_engineering
from kalman import HealthKalmanFilter
import trend_forecast
import failure_probability
import maintenance
import isolation_forest as isolation_forest_module

REPORT_PATH = config.VALIDATION_REPORT_PATH
GOLDEN_SNAPSHOT_PATH = os.path.join(config.RESULTS_DIR, "reliability_golden_snapshot.json")

# Short horizons actually testable against this dataset's ~6.9-day span —
# kept in sync with config.FAILURE_PROB_HORIZONS_DAYS so every horizon the
# pipeline actually reports gets a calibration check. 2d/3d are also
# tested here (beyond what's operational) specifically to confirm they
# stay excluded — see config.py's comment for the measured Brier scores
# that got them removed from FAILURE_PROB_HORIZONS_DAYS in the first place.
CALIBRATION_HORIZONS_DAYS = sorted(set(config.FAILURE_PROB_HORIZONS_DAYS + [2, 3]))

# ---------------------------------------------------------------------------
# Visual identity — one palette used everywhere so the console output,
# the findings, and every chart panel agree on what a color means.
# ---------------------------------------------------------------------------
COLOR = {
    "pass": "#1E8E5A",
    "warn": "#C7861B",
    "fail": "#C1443C",
    "na":   "#8A8F98",
    "normal": "#1E8E5A",
    "bad":    "#C1443C",
    "primary":   "#2456B2",
    "secondary": "#6E4DB2",
    "ink":    "#1F2430",
    "muted":  "#6B7280",
    "grid":   "#E4E7EC",
    "card_bg": "#F7F8FA",
    "panel_bg": "#FFFFFF",
}
DIMENSIONS = ["ACCURACY", "CALIBRATION", "STABILITY", "CONSISTENCY"]

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.5,
    "axes.edgecolor": COLOR["grid"],
    "axes.labelcolor": COLOR["ink"],
    "axes.titleweight": "bold",
    "axes.titlesize": 11.5,
    "text.color": COLOR["ink"],
    "xtick.color": COLOR["muted"],
    "ytick.color": COLOR["muted"],
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})


# ===========================================================================
# SCORECARD — collects every check's verdict into one trust rollup
# ===========================================================================

class Findings:
    """
    Collects each check's plain-language finding so the console log and
    the graphical report can share one source of truth. Deliberately NOT
    a scorecard: no PASS/WARN/FAIL label, no per-dimension rollup, no
    overall verdict. A single number (a threshold on train/holdout gap,
    a CI bound, a correlation cutoff) can decide which explanation is
    accurate to print, but collapsing 12 genuinely different checks —
    covering different failure modes, different statistical guarantees,
    on one 6.9-day trajectory — into one bolded verdict claims more
    certainty than any individual number here actually supports. Read
    the finding text; that's the evidence. There's no verdict above it
    to skim to instead.
    """
    def __init__(self):
        self.entries = []

    def note(self, dimension, name, detail):
        self.entries.append({"dimension": dimension, "name": name, "detail": detail})
        print(f"  • {name} — {detail}")

    def dimension_entries(self, dimension):
        return [e for e in self.entries if e["dimension"] == dimension]


# ===========================================================================
# BOOTSTRAP — block resampling so confidence intervals respect the fact
# that adjacent minutes in this data are heavily autocorrelated (same
# slowly-evolving event), not independent draws.
# ===========================================================================

def _block_index_resamples(n, block_size, n_boot, seed):
    rng = np.random.default_rng(seed)
    block_starts = list(range(0, n, block_size))
    for _ in range(n_boot):
        chosen = rng.integers(0, len(block_starts), size=len(block_starts))
        idx = np.concatenate([np.arange(bs, min(bs + block_size, n)) for bs in
                               (block_starts[c] for c in chosen)])
        yield idx[idx < n]


def bootstrap_auc_ci(y_true, scores, block_size=720, n_boot=500, seed=42):
    """95% block-bootstrap CI for ROC-AUC. Skips any resample that happens
    to contain only one class (AUC undefined) rather than crashing."""
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    stats = []
    for idx in _block_index_resamples(len(y_true), block_size, n_boot, seed):
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        stats.append(roc_auc_score(yt, scores[idx]))
    if len(stats) < 20:
        return None
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), np.array(stats)


def bootstrap_delta_auc_ci(y_true, scores_a, scores_b, block_size=720, n_boot=500, seed=42):
    """95% CI on AUC(scores_b) - AUC(scores_a), same resamples for both so
    the paired difference is honest (not two independently-noisy CIs)."""
    y_true = np.asarray(y_true)
    scores_a, scores_b = np.asarray(scores_a), np.asarray(scores_b)
    deltas = []
    for idx in _block_index_resamples(len(y_true), block_size, n_boot, seed):
        yt = y_true[idx]
        if len(np.unique(yt)) < 2:
            continue
        deltas.append(roc_auc_score(yt, scores_b[idx]) - roc_auc_score(yt, scores_a[idx]))
    if len(deltas) < 20:
        return None
    deltas = np.array(deltas)
    frac_positive = float(np.mean(deltas > 0))
    return float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5)), frac_positive


def bootstrap_brier_ci(preds, actuals, block_size=500, n_boot=500, seed=42):
    preds, actuals = np.asarray(preds), np.asarray(actuals)
    stats = []
    for idx in _block_index_resamples(len(preds), block_size, n_boot, seed):
        if len(idx) == 0:
            continue
        stats.append(float(np.mean((preds[idx] - actuals[idx]) ** 2)))
    if len(stats) < 20:
        return None
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


# ===========================================================================
# ACCURACY / DISCRIMINATION
# ===========================================================================

def check_overfitting(reference_df: pd.DataFrame, feature_cols: list, findings: Findings):
    """
    reference_df: the ACTUAL rows the currently-saved model was trained
    on (loaded via artifact_utils.load_reference_timestamps() in main()),
    not a re-derived guess — see main() for the fallback behavior when
    that tracking file is missing.
    """
    print("=" * 70)
    print("1. OVERFITTING CHECK — train/holdout split of the actual reference set")
    print("=" * 70)

    ref = reference_df
    half = len(ref) // 2
    train_half, holdout_half = ref.iloc[:half], ref.iloc[half:]

    temp_model = IsolationForest(**config.ISOLATION_FOREST_PARAMS)
    temp_model.fit(train_half[feature_cols])

    train_scores = temp_model.score_samples(train_half[feature_cols])
    holdout_scores = temp_model.score_samples(holdout_half[feature_cols])

    gap = train_scores.mean() - holdout_scores.mean()
    gap_in_std = abs(gap) / train_scores.std()

    print(f"Train-half scores   — mean: {train_scores.mean():.4f}, std: {train_scores.std():.4f}")
    print(f"Holdout-half scores — mean: {holdout_scores.mean():.4f}, std: {holdout_scores.std():.4f}")
    print(f"Mean gap: {gap:.4f} ({gap_in_std:.2f} train-std units)")

    sensor_gap_note = ""
    if gap_in_std > 1.0:
        print(
            "\n  NOTE: train/holdout gap present. Before concluding this is overfitting,\n"
            "  check whether the reference set itself is internally consistent — e.g.\n"
            "  compare raw sensor means between the two halves. Real drift already\n"
            "  present within the 'normal' rows produces this same gap and is a\n"
            "  calibration problem (reference-set selection), not overfitting."
        )
        diffs = []
        for col in config.RAW_SENSOR_COLS:
            mean_col = f"{col.split('_')[0]}_mean"
            if mean_col in ref.columns:
                t_mean, h_mean = train_half[mean_col].mean(), holdout_half[mean_col].mean()
                pct = (h_mean - t_mean) / t_mean * 100 if t_mean else float("nan")
                print(f"    {mean_col}: train-half avg={t_mean:.3f}, holdout-half avg={h_mean:.3f} ({pct:+.1f}%)")
                diffs.append(abs(pct))
        if diffs:
            sensor_gap_note = f" Largest raw-sensor drift between halves: {max(diffs):.1f}%."
    print()

    if gap_in_std <= 1.0:
        findings.note("ACCURACY", "Overfitting check",
                                f"train/holdout gap is {gap_in_std:.2f} train-std units — no meaningful signal of overfitting.")
    elif gap_in_std <= 3.0:
        findings.note("ACCURACY", "Overfitting check",
                                f"train/holdout gap is {gap_in_std:.2f} train-std units.{sensor_gap_note} Likely reference-window non-stationarity rather than pure overfitting — see per-sensor breakdown above.")
    else:
        findings.note("ACCURACY", "Overfitting check",
                                f"train/holdout gap is a large {gap_in_std:.2f} train-std units.{sensor_gap_note} Investigate before trusting the reference calibration.")
    return train_scores, holdout_scores


# ===========================================================================
# FULL BACKTEST — vectorized scoring, sequential Kalman/trend/maintenance
# ===========================================================================

def run_backtest(df: pd.DataFrame, feature_cols: list, scorer):
    """
    Reruns the full pipeline (minus feature engineering, which is assumed
    already applied to df) exactly the way predict_realtime.SpindleMonitor
    would, tick by tick, but with vectorized anomaly scoring up front for
    speed. Used as the shared backbone for the accuracy/calibration checks
    AND reused (on perturbed or sliced inputs) by the stability and
    consistency checks below — one implementation, several tests run
    against it.
    """
    raw_scores = scorer.score(df[feature_cols])
    health_raw = scorer.health_from_score(raw_scores)

    n_init = config.KALMAN_INIT_SAMPLES
    if len(health_raw) <= n_init:
        raise ValueError(
            f"Need more than KALMAN_INIT_SAMPLES ({n_init}) rows to run a backtest."
        )
    initial_level = float(np.mean(health_raw[:n_init]))
    kalman = HealthKalmanFilter(initial_level=initial_level)
    maintenance_debouncer = maintenance.MaintenanceDebouncer()

    health_states, maint_levels = [], []
    slopes, residuals = [], []
    trend_minutes, trend_health = [], []
    trend_fit_ticks = 0

    for i, h in enumerate(health_raw):
        if i < n_init:
            continue

        hs = kalman.update(h)
        health_states.append(hs)
        trend_minutes.append(i)
        trend_health.append(hs)
        if len(trend_minutes) > config.TREND_LOOKBACK_MINUTES:
            trend_minutes.pop(0)
            trend_health.pop(0)

        fit = trend_forecast.fit_trend(np.array(trend_minutes), np.array(trend_health))
        if fit is None:
            maint_levels.append("OK")
            slopes.append(np.nan)
            residuals.append(np.nan)
            continue

        trend_fit_ticks += 1
        slope, intercept, resid = fit
        settled = trend_fit_ticks >= config.TREND_SETTLE_TICKS
        significant = trend_forecast.slope_is_significant(np.array(trend_minutes), slope, resid)
        trend_trusted = settled and significant

        slopes.append(slope)
        residuals.append(resid)
        remaining = trend_forecast.remaining_days(i, hs, slope)
        prob_table = failure_probability.failure_probability_table(hs, slope, resid)
        rec = maintenance_debouncer.evaluate(hs, remaining, prob_table, trend_trusted=trend_trusted)
        maint_levels.append(rec["level"])

    out = df.iloc[n_init:][[config.COL_TIMESTAMP]].copy()
    out["health_raw"] = health_raw[n_init:]
    out["health_state"] = health_states
    out["maintenance_level"] = maint_levels
    out["slope_per_minute"] = slopes
    out["residual_std"] = residuals
    return out


def build_holdout_merged(scorer, feature_cols: list):
    """
    Replays config.PREDICT_DATA_PATH — a file the reference set was never
    drawn from — through the same run_backtest() used on the training
    file, and merges the result with ITS OWN health_status labels.
    Returns (merged, y_true, is_bad_array, backtest_df) in the exact
    shape check_discrimination()/check_calibration()/
    check_threshold_calibration() already expect, so main() can hand
    them this instead of the training-file version with no changes to
    those functions — same statistics, different (and, right now, the
    only usable) data source. Returns None if PREDICT_DATA_PATH has no
    health_status column.
    """
    holdout_raw = pd.read_csv(config.PREDICT_DATA_PATH)
    if "health_status" not in holdout_raw.columns:
        return None

    holdout_clean = preprocessing.clean_data(
        holdout_raw[[config.COL_TIMESTAMP] + config.RAW_SENSOR_COLS].copy())
    holdout_features = feature_engineering.create_features(holdout_clean, verbose=False)
    holdout_backtest = run_backtest(holdout_features, feature_cols, scorer)

    labels = holdout_raw[[config.COL_TIMESTAMP, "health_status"]].copy()
    labels[config.COL_TIMESTAMP] = pd.to_datetime(labels[config.COL_TIMESTAMP])
    labels = labels.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
    labels["pos"] = np.arange(len(labels))
    labels["is_bad"] = (labels["health_status"] != "normal").astype(int)

    merged = holdout_backtest.merge(labels[[config.COL_TIMESTAMP, "health_status", "pos"]],
                                     on=config.COL_TIMESTAMP, how="left")
    y_true = (merged["health_status"] != "normal").astype(int).values
    is_bad_array = labels["is_bad"].values
    return merged, y_true, is_bad_array, holdout_backtest


def confusion(pred: np.ndarray, y_true: np.ndarray, label: str):
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")

    print(f"--- {label} ---")
    print(f"  True Positive  (correctly flagged bad):      {tp}")
    print(f"  False Negative (missed a real problem):      {fn}")
    print(f"  False Positive (flagged bad, actually fine): {fp}")
    print(f"  True Negative  (correctly left alone):       {tn}")
    print(f"  Precision: {precision:.3f}   Recall: {recall:.3f}")
    print()
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn, "precision": precision, "recall": recall}


def check_discrimination(merged: pd.DataFrame, y_true: np.ndarray, findings: Findings):
    print("=" * 70)
    print("2. DISCRIMINATION — ROC-AUC, PR-AUC, threshold sweep, bootstrap CIs")
    print("=" * 70)

    health_raw_neg = -merged["health_raw"].values
    health_state_neg = -merged["health_state"].values

    auc_raw = roc_auc_score(y_true, health_raw_neg)
    auc_state = roc_auc_score(y_true, health_state_neg)
    ap_state = average_precision_score(y_true, health_state_neg)
    print(f"ROC-AUC — health_raw (pre-Kalman):          {auc_raw:.4f}")
    print(f"ROC-AUC — health_state (post-Kalman+trend): {auc_state:.4f}")
    print(f"PR-AUC (average precision) — health_state:  {ap_state:.4f}")

    ci_raw = bootstrap_auc_ci(y_true, health_raw_neg)
    ci_state = bootstrap_auc_ci(y_true, health_state_neg)
    delta = bootstrap_delta_auc_ci(y_true, health_raw_neg, health_state_neg)
    if ci_raw:
        print(f"  95% block-bootstrap CI — AUC health_raw:   [{ci_raw[0]:.4f}, {ci_raw[1]:.4f}]")
    if ci_state:
        print(f"  95% block-bootstrap CI — AUC health_state: [{ci_state[0]:.4f}, {ci_state[1]:.4f}]")
    if delta:
        print(f"  95% CI on (AUC_state - AUC_raw): [{delta[0]:+.4f}, {delta[1]:+.4f}]  "
              f"({delta[2]:.0%} of resamples favor health_state)")
    print()

    print("Threshold sweep (health_state <= X):")
    print(f"{'threshold':>10} | {'TPR':>6} | {'FPR':>6}")
    for thresh in [80, 60, 40, 20, 10]:
        pred = (merged["health_state"] <= thresh).astype(int).values
        tp = ((pred == 1) & (y_true == 1)).sum()
        fn = ((pred == 0) & (y_true == 1)).sum()
        fp = ((pred == 1) & (y_true == 0)).sum()
        tn = ((pred == 0) & (y_true == 0)).sum()
        tpr = tp / (tp + fn) if (tp + fn) else float("nan")
        fpr = fp / (fp + tn) if (fp + tn) else float("nan")
        print(f"{thresh:>10} | {tpr:>6.3f} | {fpr:>6.3f}")
    print()

    print("=" * 70)
    print("3. CONFUSION MATRICES")
    print("=" * 70)
    print(f"Total rows: {len(merged)}  |  Actually bad: {y_true.sum()}  |  Actually normal: {(y_true==0).sum()}")
    print()

    pred_threshold = (merged["health_state"] <= config.FAILURE_HEALTH_THRESHOLD).astype(int).values
    conf_threshold = confusion(pred_threshold, y_true, f"health_state <= {config.FAILURE_HEALTH_THRESHOLD} (raw threshold only)")

    pred_maint = (merged["maintenance_level"] == "CRITICAL").astype(int).values
    conf_maint = confusion(pred_maint, y_true, "maintenance_level == CRITICAL (full pipeline output)")

    # --- findings: discrimination strength, using the LOWER bound of the
    # bootstrap CI (not the point estimate) so the verdict already accounts
    # for the single-trajectory uncertainty rather than needing the reader
    # to mentally discount it.
    if ci_state:
        lower = ci_state[0]
        if lower >= 0.90:
            findings.note("ACCURACY", "Discrimination (AUC, health_state)",
                          f"AUC={auc_state:.3f}, 95% CI lower bound {lower:.3f} — strong separation even under the pessimistic end of the bootstrap.")
        elif lower >= 0.75:
            findings.note("ACCURACY", "Discrimination (AUC, health_state)",
                          f"AUC={auc_state:.3f}, but 95% CI lower bound is only {lower:.3f} — decent point estimate, less certain given one trajectory.")
        else:
            findings.note("ACCURACY", "Discrimination (AUC, health_state)",
                          f"AUC={auc_state:.3f}, 95% CI lower bound {lower:.3f} — not reliably better than chance under resampling.")
    else:
        findings.note("ACCURACY", "Discrimination (AUC, health_state)",
                      f"AUC={auc_state:.3f} — too few valid bootstrap resamples to form a CI; treat the point estimate cautiously.")

    # --- findings: does the Kalman+trend layer actually help, or could
    # the apparent improvement be noise from one trajectory?
    if delta:
        lo, hi, frac_pos = delta
        if lo > 0:
            findings.note("ACCURACY", "Kalman+trend adds value",
                          f"95% CI on the AUC improvement is [{lo:+.3f}, {hi:+.3f}] — entirely positive, improvement looks real.")
        elif hi < 0:
            findings.note("ACCURACY", "Kalman+trend adds value",
                          f"95% CI on the AUC improvement is [{lo:+.3f}, {hi:+.3f}] — entirely negative, the smoothing layer appears to hurt discrimination here.")
        else:
            findings.note("ACCURACY", "Kalman+trend adds value",
                          f"95% CI on the AUC improvement straddles zero [{lo:+.3f}, {hi:+.3f}] ({frac_pos:.0%} of resamples positive) — inconclusive from one trajectory.")

    return auc_raw, auc_state, ap_state, conf_threshold, conf_maint, ci_raw, ci_state


# ===========================================================================
# CALIBRATION / RELIABILITY
# ===========================================================================

def check_calibration(merged: pd.DataFrame, is_bad_array: np.ndarray, findings: Findings):
    print("=" * 70)
    print("4 & 5. CALIBRATION — Brier score and reliability diagrams")
    print("=" * 70)
    print(
        "Checks whether failure_probability_table()'s numbers mean what they\n"
        "claim: among all times the model said '70% chance', did the bad state\n"
        "actually occur about 70% of the time? Different property from\n"
        "discrimination (AUC) — a model can rank cases correctly while still\n"
        "being poorly calibrated.\n"
    )
    print(
        f"CAVEAT: this dataset spans ~{len(is_bad_array)/1440:.1f} days with exactly ONE\n"
        f"escalation event. config.FAILURE_PROB_HORIZONS_DAYS "
        f"({config.FAILURE_PROB_HORIZONS_DAYS}) were restricted to <=1 day\n"
        f"specifically because longer horizons (2d/3d, tested below for reference)\n"
        f"measured Brier scores worse than the uninformative baseline. Treat this\n"
        f"as a first look, not a robust calibration guarantee.\n"
    )

    n_total = len(is_bad_array)
    results = {}
    operational_horizon_statuses = []

    for h in CALIBRATION_HORIZONS_DAYS:
        h_minutes = int(h * 24 * 60)
        preds, actuals = [], []

        for idx, row in merged.iterrows():
            if pd.isna(row["slope_per_minute"]):
                continue
            pos = row["pos"]
            window_end = pos + h_minutes
            if window_end >= n_total:
                continue

            actual = int(is_bad_array[pos + 1: window_end + 1].max())
            pred = failure_probability.failure_probability_at(
                row["health_state"], row["slope_per_minute"], row["residual_std"], h
            )
            preds.append(pred)
            actuals.append(actual)

        if len(preds) < 30:
            print(f"Horizon {h}d: only {len(preds)} evaluable rows — too few to report, skipped.")
            continue

        preds, actuals = np.array(preds), np.array(actuals)
        brier = float(np.mean((preds - actuals) ** 2))
        ci = bootstrap_brier_ci(preds, actuals)
        ci_str = f", 95% CI [{ci[0]:.4f}, {ci[1]:.4f}]" if ci else ""
        print(f"Horizon {h}d: n={len(preds)}, Brier score={brier:.4f}{ci_str} "
              f"(0=perfect, 0.25=uninformative-constant-0.5, 1=worst)")

        bins = np.linspace(0, 1, 6)
        bin_idx = np.digitize(preds, bins) - 1
        bin_idx = np.clip(bin_idx, 0, len(bins) - 2)
        bin_pred, bin_actual, bin_n = [], [], []
        for b in range(len(bins) - 1):
            mask = bin_idx == b
            if mask.sum() == 0:
                continue
            bin_pred.append(preds[mask].mean())
            bin_actual.append(actuals[mask].mean())
            bin_n.append(int(mask.sum()))
        print(f"  Reliability bins — predicted vs observed: "
              + ", ".join(f"({p:.2f}->{a:.2f}, n={n})" for p, a, n in zip(bin_pred, bin_actual, bin_n)))

        results[h] = {"preds": preds, "actuals": actuals, "brier": brier, "ci": ci,
                      "bin_pred": bin_pred, "bin_actual": bin_actual, "bin_n": bin_n}

        if h in config.FAILURE_PROB_HORIZONS_DAYS:
            upper = ci[1] if ci else brier
            if upper < 0.15:
                operational_horizon_statuses.append("PASS")
            elif upper < 0.25:
                operational_horizon_statuses.append("WARN")
            else:
                operational_horizon_statuses.append("FAIL")

    for h in config.FAILURE_PROB_HORIZONS_DAYS:
        if h not in CALIBRATION_HORIZONS_DAYS:
            h_minutes = int(h * 24 * 60)
            evaluable = int((merged["pos"] + h_minutes < n_total).sum())
            print(f"Horizon {h}d (operational, not calibration-tested): "
                  f"only {evaluable} rows would have complete future data — skipped.")
    print()

    if not operational_horizon_statuses:
        findings.note("CALIBRATION", "Failure-probability calibration (operational horizons)",
                      "no operational horizon had enough evaluable rows to score.")
    elif all(s == "PASS" for s in operational_horizon_statuses):
        best = max(results[h]["brier"] for h in config.FAILURE_PROB_HORIZONS_DAYS if h in results)
        findings.note("CALIBRATION", "Failure-probability calibration (operational horizons)",
                      f"every operational horizon's 95% CI upper bound stays comfortably below the 0.25 uninformative baseline (worst Brier point estimate {best:.3f}).")
    elif "FAIL" in operational_horizon_statuses:
        findings.note("CALIBRATION", "Failure-probability calibration (operational horizons)",
                      "at least one operational horizon's Brier CI upper bound reaches the uninformative baseline (0.25) or worse.")
    else:
        findings.note("CALIBRATION", "Failure-probability calibration (operational horizons)",
                      "operational horizons beat guessing on the point estimate, but the bootstrap CI's upper bound isn't comfortably clear of 0.25.")

    beyond = [h for h in CALIBRATION_HORIZONS_DAYS if h not in config.FAILURE_PROB_HORIZONS_DAYS and h in results]
    if beyond:
        worst_beyond = max(results[h]["brier"] for h in beyond)
        if worst_beyond >= 0.25:
            findings.note("CALIBRATION", "Extended horizons correctly excluded",
                          f"horizon(s) {beyond} score worse than or near the uninformative baseline ({worst_beyond:.3f}) and are correctly left out of FAILURE_PROB_HORIZONS_DAYS.")
        else:
            findings.note("CALIBRATION", "Extended horizons correctly excluded",
                          f"horizon(s) {beyond} now score {worst_beyond:.3f} (below the 0.25 baseline) — config.py's exclusion of these may be stale; worth re-checking.")

    return results


# ===========================================================================
# STABILITY
# ===========================================================================

def check_perturbation_robustness(df: pd.DataFrame, feature_cols: list, scorer,
                                   baseline_bt: pd.DataFrame, findings: Findings,
                                   noise_levels=(0.01, 0.03, 0.05), n_repeats=2, seed=42):
    """
    Perturbs every engineered FEATURE (not the raw sensor columns) by
    Gaussian noise scaled to a fraction of that feature's own std,
    reruns scoring + Kalman + trend + maintenance, and measures how much
    health_state / maintenance_level move relative to the unperturbed
    backtest. Perturbing at the feature level rather than re-running
    feature_engineering.py on noised raw sensors is a deliberate cost
    tradeoff (rolling kurtosis/skew over the full ~10k-row trajectory is
    the slow step in this whole suite) — it still directly tests what the
    scorer actually consumes, it just doesn't test whether the rolling-
    window computation itself amplifies raw sensor noise. That's a real,
    narrower scope than a raw-sensor perturbation test would have; note
    it rather than claim more than this measures.
    """
    print("=" * 70)
    print("6. PERTURBATION ROBUSTNESS — feature-level noise injection")
    print("=" * 70)

    rng = np.random.default_rng(seed)
    feat_std = df[feature_cols].std().replace(0, np.nan).fillna(1e-6).values
    baseline_health = baseline_bt["health_state"].values
    baseline_maint = baseline_bt["maintenance_level"].values

    results = {}
    for level in noise_levels:
        corrs, mads, flips = [], [], []
        for _ in range(n_repeats):
            noisy_df = df.copy()
            noise = rng.normal(0, 1, size=df[feature_cols].shape) * (feat_std * level)
            noisy_df[feature_cols] = df[feature_cols].values + noise
            bt = run_backtest(noisy_df, feature_cols, scorer)
            hs = bt["health_state"].values
            n = min(len(hs), len(baseline_health))
            corr = float(np.corrcoef(hs[:n], baseline_health[:n])[0, 1])
            mad = float(np.mean(np.abs(hs[:n] - baseline_health[:n])))
            flip = float(np.mean(bt["maintenance_level"].values[:n] != baseline_maint[:n]))
            corrs.append(corr); mads.append(mad); flips.append(flip)
        results[level] = {
            "corr_mean": float(np.mean(corrs)), "corr_min": float(np.min(corrs)),
            "mad_mean": float(np.mean(mads)), "flip_rate_mean": float(np.mean(flips)),
        }
        print(f"  noise = {level:.0%} of each feature's std -> "
              f"health_state corr={np.mean(corrs):.4f} (min {np.min(corrs):.4f} across {n_repeats} reps), "
              f"mean|Δhealth_state|={np.mean(mads):.2f}pp, "
              f"maintenance_level flip rate={np.mean(flips):.2%}")
    print()

    worst_corr = min(r["corr_min"] for r in results.values())
    worst_flip = max(r["flip_rate_mean"] for r in results.values())
    max_level = max(noise_levels)
    if worst_corr >= 0.99 and worst_flip <= 0.02:
        findings.note("STABILITY", "Perturbation robustness",
                      f"health_state stays highly correlated (worst r={worst_corr:.4f}) and maintenance_level rarely flips (worst {worst_flip:.1%}) under up to {max_level:.0%} feature-level noise.")
    elif worst_corr >= 0.95 and worst_flip <= 0.10:
        findings.note("STABILITY", "Perturbation robustness",
                      f"noticeable sensitivity at higher noise levels (worst r={worst_corr:.4f}, flip rate={worst_flip:.1%}) — borderline readings may not be robust to measurement noise.")
    else:
        findings.note("STABILITY", "Perturbation robustness",
                      f"output is materially unstable under small feature-level noise (worst r={worst_corr:.4f}, flip rate={worst_flip:.1%}).")
    return results


def check_fault_injection(df: pd.DataFrame, feature_cols: list, scorer, findings: Findings):
    """
    Feeds the trained scorer deliberately extreme or degenerate feature
    rows it will never see from clean rolling-window output, and checks
    it either (a) degrades to a bounded, sane health reading, or (b)
    fails loudly and immediately — never (c) silently returns a
    plausible-looking but meaningless number. (c) is the dangerous case
    for a monitoring system: a silent wrong answer is worse than a
    crash, because nothing downstream knows to distrust it.
    """
    print("=" * 70)
    print("7. FAULT INJECTION — extreme, degenerate, and missing inputs")
    print("=" * 70)

    baseline_row = df[feature_cols].iloc[[0]].copy()
    cases = {}

    extreme = baseline_row.copy()
    extreme.loc[:, :] = baseline_row.values * 1000 + 1e6
    cases["extreme outlier (1000x + 1e6 offset)"] = extreme

    zero = baseline_row.copy()
    zero.loc[:, :] = 0.0
    cases["all-zero feature vector"] = zero

    if scorer.baseline_feature_mean is not None:
        mean_row = baseline_row.copy()
        mean_row.loc[:, :] = scorer.baseline_feature_mean.reindex(feature_cols).values.reshape(1, -1)
        cases["exact reference-baseline mean"] = mean_row

    all_ok = True
    for name, row in cases.items():
        try:
            score = scorer.score(row)
            health = float(scorer.health_from_score(score)[0])
            bounded = np.isfinite(health) and 0.0 <= health <= 100.0
            print(f"  {name}: health={health:.2f}%  -> {'bounded, OK' if bounded else 'OUT OF BOUNDS'}")
            if not bounded:
                all_ok = False
        except Exception as e:
            print(f"  {name}: raised {type(e).__name__} ({e}) — loud failure, acceptable.")

    nan_row = baseline_row.copy()
    nan_row.iloc[0, 0] = np.nan
    try:
        score = scorer.score(nan_row)
        health = float(scorer.health_from_score(score)[0])
        print(f"  NaN in one feature: silently returned health={health:.2f}% — NOT caught.")
        nan_caught = False
    except Exception as e:
        print(f"  NaN in one feature: raised {type(e).__name__} as expected (loud failure, correct behavior).")
        nan_caught = True
    print()

    if all_ok and nan_caught:
        findings.note("STABILITY", "Fault injection",
                      "extreme/degenerate inputs stay bounded in [0,100]%, and a NaN feature fails loudly instead of silently producing a number.")
    elif all_ok and not nan_caught:
        findings.note("STABILITY", "Fault injection",
                      "extreme/degenerate inputs stay bounded, but a NaN feature silently produced a health reading instead of failing loudly — a real sensor dropout could masquerade as a valid score.")
    else:
        findings.note("STABILITY", "Fault injection",
                      "at least one extreme/degenerate input produced an out-of-bounds or non-finite health reading.")


def check_reference_window_sensitivity(df: pd.DataFrame, feature_cols: list, original_health_raw: np.ndarray,
                                        reference_df: pd.DataFrame, findings: Findings,
                                        shifts_minutes=(-360, -120, 120, 360)):
    """
    Refits a fresh IsolationForest on reference windows shifted earlier/
    later by a few hours from the window the shipped model actually used
    (same length, same hyperparameters), and compares the resulting
    health_raw trajectory (vectorized score on the full feature set)
    against the shipped model's. Tests sensitivity to *exactly* where the
    commissioning/burn-in window was drawn — a different question from
    whether that window was the "right" one in the first place (see the
    overfitting check above for that).
    """
    print("=" * 70)
    print("8. REFERENCE-WINDOW SENSITIVITY — refit on nearby burn-in windows")
    print("=" * 70)

    ref_len = len(reference_df)
    start_ts = reference_df[config.COL_TIMESTAMP].min()
    start_idx = int(df[config.COL_TIMESTAMP].searchsorted(start_ts))

    results = {}
    for shift in shifts_minutes:
        new_start = int(np.clip(start_idx + shift, 0, max(0, len(df) - ref_len)))
        shifted_ref = df.iloc[new_start:new_start + ref_len]
        if len(shifted_ref) < ref_len // 2:
            continue
        shifted_scorer = isolation_forest_module.AnomalyScorer()
        shifted_scorer.fit(shifted_ref[feature_cols])
        shifted_scores = shifted_scorer.score(df[feature_cols])
        shifted_health = shifted_scorer.health_from_score(shifted_scores)

        corr = float(np.corrcoef(shifted_health, original_health_raw)[0, 1])
        mad = float(np.mean(np.abs(shifted_health - original_health_raw)))
        results[shift] = {"corr": corr, "mad": mad}
        print(f"  shift {shift:+d} min -> health_raw corr={corr:.4f}, mean|Δ|={mad:.2f}pp vs. shipped model")
    print()

    if not results:
        findings.note("STABILITY", "Reference-window sensitivity",
                      "no valid shifted window could be constructed (dataset too short for the tested shifts).")
        return results

    worst_corr = min(r["corr"] for r in results.values())
    worst_mad = max(r["mad"] for r in results.values())
    if worst_corr >= 0.98:
        findings.note("STABILITY", "Reference-window sensitivity",
                      f"health_raw stays nearly identical (worst corr={worst_corr:.4f}, worst mean|Δ|={worst_mad:.2f}pp) when the burn-in window is nudged by up to {max(abs(s) for s in shifts_minutes)} min.")
    elif worst_corr >= 0.90:
        findings.note("STABILITY", "Reference-window sensitivity",
                      f"moderate sensitivity to exactly where the burn-in window starts (worst corr={worst_corr:.4f}, worst mean|Δ|={worst_mad:.2f}pp).")
    else:
        findings.note("STABILITY", "Reference-window sensitivity",
                      f"health readings are highly sensitive to the burn-in window's exact position (worst corr={worst_corr:.4f}) — calibration is fragile.")
    return results


# ===========================================================================
# CONSISTENCY
# ===========================================================================

def check_reproducibility(reference_df: pd.DataFrame, feature_cols: list, findings: Findings):
    """
    Refits IsolationForest twice on identical reference data with the
    identical config (including the fixed random_state=42) and confirms
    the two fits score identically. The minimum bar for "consistency":
    if this fails, nothing downstream can be trusted to reproduce either.
    """
    print("=" * 70)
    print("9. TRAINING REPRODUCIBILITY — two fits, identical data")
    print("=" * 70)

    m1 = IsolationForest(**config.ISOLATION_FOREST_PARAMS).fit(reference_df[feature_cols])
    m2 = IsolationForest(**config.ISOLATION_FOREST_PARAMS).fit(reference_df[feature_cols])
    s1 = m1.score_samples(reference_df[feature_cols])
    s2 = m2.score_samples(reference_df[feature_cols])
    max_diff = float(np.max(np.abs(s1 - s2)))
    exact = bool(np.array_equal(s1, s2))
    print(f"  max |score difference| between two independent fits: {max_diff:.2e} (exact match: {exact})")
    print()

    if exact or max_diff < 1e-9:
        findings.note("CONSISTENCY", "Training reproducibility",
                      f"two independent fits on identical data match to within floating-point noise (max diff {max_diff:.2e}) — random_state is doing its job.")
    else:
        findings.note("CONSISTENCY", "Training reproducibility",
                      f"two independent fits on identical data diverge (max diff {max_diff:.4f}) despite a fixed random_state.")
    return {"max_diff": max_diff, "exact": exact}


def check_batch_vs_realtime(feature_cols: list, findings: Findings, n_ticks: int = 700):
    """
    Runs predict_realtime.py's ACTUAL SpindleMonitor tick-by-tick over the
    first n_ticks raw sensor rows, and compares its output to this
    script's vectorized run_backtest() over the equivalent slice of the
    already-computed feature set. These are two independently-written
    code paths meant to compute the same thing — this is the test that
    checks that claim directly, instead of relying on someone reading
    both implementations carefully. n_ticks is capped well below the full
    dataset because SpindleMonitor recomputes rolling features from
    scratch every tick (by design, see its module docstring) and isn't
    fast enough to replay ~10k rows here.
    """
    print("=" * 70)
    print("10. BATCH vs. REAL-TIME EQUIVALENCE")
    print("=" * 70)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with contextlib.redirect_stdout(io.StringIO()):
            import predict_realtime

    raw = pd.read_csv(config.RAW_DATA_PATH, usecols=[config.COL_TIMESTAMP] + config.RAW_SENSOR_COLS).iloc[:n_ticks]
    monitor = predict_realtime.SpindleMonitor()
    live_rows = []
    for _, row in raw.iterrows():
        reading = {
            config.COL_VIBRATION: float(row[config.COL_VIBRATION]),
            config.COL_TEMPERATURE: float(row[config.COL_TEMPERATURE]),
            config.COL_CURRENT: float(row[config.COL_CURRENT]),
        }
        res = monitor.update(reading)
        if res is not None:
            live_rows.append({
                "health_raw": res["health_raw"], "health_state": res["health_state"],
                "maintenance_level": res["maintenance"]["level"],
            })
    live_df = pd.DataFrame(live_rows)

    scorer, feats, _ = artifact_utils.load_artifacts()
    features_full = pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])
    k = max(config.KALMAN_INIT_SAMPLES + 1, n_ticks - config.WINDOW_SIZE + 1)
    batch_slice = features_full.head(min(k, len(features_full)))
    batch_bt = run_backtest(batch_slice, feats, scorer)

    # Known, documented one-row offset: SpindleMonitor's FIRST non-None
    # result is the just-seeded Kalman level itself (mean of the init
    # buffer, no update() applied yet); run_backtest's health_states only
    # starts from the first kalman.update() call, one tick later. So the
    # real-time path always emits exactly one more leading reading than
    # the vectorized backtest for the same input. That extra row is
    # dropped here so the two are compared tick-for-tick on the readings
    # they both actually produce — this offset itself is reported as its
    # own finding below, not silently discarded.
    seed_row_offset = 1 if len(live_df) == len(batch_bt) + 1 else 0
    live_compare = live_df.iloc[seed_row_offset:].reset_index(drop=True)

    n = min(len(live_compare), len(batch_bt))
    print(f"  real-time path emitted {len(live_df)} readings, batch path emitted {len(batch_bt)} "
          f"(real-time's first reading is its Kalman-seed row, which the batch path never "
          f"emits by design — dropped before comparing; comparing {n} tick-aligned readings)")

    if n == 0:
        findings.note("CONSISTENCY", "Batch vs. real-time equivalence",
                      "one of the two code paths emitted zero readings — cannot compare.")
        return {"n_compared": 0}

    live_hs = live_compare["health_state"].values[:n]
    batch_hs = batch_bt["health_state"].values[:n]
    live_hr = live_compare["health_raw"].values[:n]
    batch_hr = batch_bt["health_raw"].values[:n]
    max_diff_state = float(np.max(np.abs(live_hs - batch_hs)))
    max_diff_raw = float(np.max(np.abs(live_hr - batch_hr)))
    maint_match = float(np.mean(live_compare["maintenance_level"].values[:n] == batch_bt["maintenance_level"].values[:n]))
    len_match = len(live_compare) == len(batch_bt)

    print(f"  max|Δ health_raw|={max_diff_raw:.4f}pp   max|Δ health_state|={max_diff_state:.4f}pp   "
          f"maintenance_level agreement={maint_match:.2%}   tick-aligned length match={len_match}")
    print()

    seed_note = "" if seed_row_offset else " (NOTE: lengths didn't differ by the expected 1-row seed offset — investigate before trusting this comparison.)"
    if len_match and max_diff_state < 0.5 and maint_match == 1.0:
        findings.note("CONSISTENCY", "Batch vs. real-time equivalence",
                      f"the two independent code paths agree almost exactly over {n} tick-aligned readings (max|Δhealth_state|={max_diff_state:.3f}pp, 100% maintenance_level agreement).{seed_note}")
    elif max_diff_state < 2.0 and maint_match >= 0.98:
        findings.note("CONSISTENCY", "Batch vs. real-time equivalence",
                      f"small divergence between the two code paths (max|Δhealth_state|={max_diff_state:.3f}pp, {maint_match:.1%} maintenance_level agreement).{seed_note}")
    else:
        findings.note("CONSISTENCY", "Batch vs. real-time equivalence",
                      f"the vectorized backtest and the real production loop disagree materially (max|Δhealth_state|={max_diff_state:.3f}pp, {maint_match:.1%} maintenance_level agreement).{seed_note} One of the two implementations has a bug relative to the other.")

    if seed_row_offset:
        findings.note("CONSISTENCY", "Real-time emits an extra seed reading",
                      "predict_realtime.SpindleMonitor's first non-None result is the just-seeded Kalman level (no update() applied yet); run_backtest never emits that row — real-time output has one extra leading reading vs. the vectorized backtest for the same input. Minor, but worth knowing if row counts are compared downstream.")
    return {"n_compared": n, "max_diff_state": max_diff_state, "maint_match": maint_match, "len_match": len_match}


def check_golden_snapshot(backtest_df: pd.DataFrame, findings: Findings, n_rows: int = 500):
    """
    Hashes a fixed, rounded slice of THIS run's backtest output and
    compares it against a snapshot saved by a previous run of this
    script (results/reliability_golden_snapshot.json). This is the only
    check in the suite that looks backward across runs rather than
    within a single run — it's what catches "someone changed the code
    and the output silently changed" between retrains. First run
    establishes the baseline. A mismatch is reported as WARN, not FAIL:
    a deliberate change is supposed to move these numbers, so treat a
    mismatch as "go confirm this was intentional," not an automatic
    alarm.
    """
    print("=" * 70)
    print("11. CROSS-RUN REGRESSION — golden snapshot")
    print("=" * 70)

    n_rows = min(n_rows, len(backtest_df))
    slice_df = backtest_df.head(n_rows)[["health_raw", "health_state", "maintenance_level"]].copy()
    slice_df["health_raw"] = slice_df["health_raw"].round(3)
    slice_df["health_state"] = slice_df["health_state"].round(3)
    payload = {k: list(v) for k, v in slice_df.to_dict(orient="list").items()}
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    if os.path.exists(GOLDEN_SNAPSHOT_PATH):
        with open(GOLDEN_SNAPSHOT_PATH) as f:
            golden = json.load(f)
        if golden.get("hash") == digest:
            print(f"  MATCH — identical to the baseline saved {golden.get('saved_at', '?')}.")
            findings.note("CONSISTENCY", "Cross-run regression (golden snapshot)",
                          f"output over the first {n_rows} post-warm-up ticks matches the baseline saved {golden.get('saved_at', '?')} exactly.")
        else:
            old_hs = np.array(golden.get("payload", {}).get("health_state", []))
            new_hs = slice_df["health_state"].values
            n_diff = int(np.sum(np.abs(old_hs - new_hs) > 1e-6)) if len(old_hs) == len(new_hs) else n_rows
            print(f"  MISMATCH vs. baseline saved {golden.get('saved_at', '?')} — {n_diff}/{n_rows} rows differ in health_state.")
            print("  Baseline NOT auto-updated — delete the snapshot file to accept this as the new baseline.")
            findings.note("CONSISTENCY", "Cross-run regression (golden snapshot)",
                          f"output changed vs. the baseline saved {golden.get('saved_at', '?')} ({n_diff}/{n_rows} rows differ) — confirm this was an intended change.")
    else:
        with open(GOLDEN_SNAPSHOT_PATH, "w") as f:
            json.dump({"hash": digest, "saved_at": pd.Timestamp.now().isoformat(), "payload": payload}, f)
        print(f"  No prior baseline found — saved this run as the new baseline -> {GOLDEN_SNAPSHOT_PATH}")
        findings.note("CONSISTENCY", "Cross-run regression (golden snapshot)",
                      f"no prior baseline existed — saved this run's first {n_rows} ticks as the baseline for future regression checks.")
    print()


# ===========================================================================
# THRESHOLD SANITY CHECK — informational only (see docstring for why this
# can't be a hard PASS/FAIL the way the checks above are)
# ===========================================================================

def check_threshold_calibration(merged: pd.DataFrame, findings: Findings, raw_data_path: str = None):
    """
    Sanity-checks config.FAILURE_HEALTH_THRESHOLD (20) and
    config.MAINTENANCE_HEALTH_INSPECT (40) against an empirically-derived
    "onset" per sensor: the midpoint between each health_status class's
    IQR edges. This is a genuinely different, WEAKER kind of evidence
    than an independent physical spec bound — it's "where the labels
    already say severity changes," not an outside reference — so unlike
    every other check in this file, it is deliberately NOT rolled into
    the PASS/FAIL findings as a hard gate. It's recorded as informational
    context only (dimension CONSISTENCY, always WARN-or-better) so it's
    still visible in the report without implying it carries the same
    evidentiary weight as, say, a bootstrapped AUC.
    """
    print("=" * 70)
    print("12. THRESHOLD SANITY CHECK (informational) — health% thresholds vs. sensor onset")
    print("=" * 70)

    raw = pd.read_csv(raw_data_path or config.RAW_DATA_PATH)
    raw[config.COL_TIMESTAMP] = pd.to_datetime(raw[config.COL_TIMESTAMP])
    raw = raw.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)

    iqr = raw.groupby("health_status")[config.RAW_SENSOR_COLS].quantile([0.25, 0.75]).unstack()
    missing = [s for s in ("normal", "warning", "critical") if s not in iqr.index]
    if missing:
        print(f"  Skipped: health_status is missing classes {missing} in this dataset.\n")
        findings.note("CONSISTENCY", "Threshold sanity check (informational)",
                      f"skipped — health_status is missing class(es) {missing} in this dataset.")
        return None

    warn_onset = {c: (iqr.loc["normal", (c, 0.75)] + iqr.loc["warning", (c, 0.25)]) / 2
                  for c in config.RAW_SENSOR_COLS}
    crit_onset = {c: (iqr.loc["warning", (c, 0.75)] + iqr.loc["critical", (c, 0.25)]) / 2
                  for c in config.RAW_SENSOR_COLS}
    print(f"  Empirical warning onset (any sensor >=): "
          f"{ {k: round(v, 2) for k, v in warn_onset.items()} }")
    print(f"  Empirical critical onset (any sensor >=): "
          f"{ {k: round(v, 2) for k, v in crit_onset.items()} }")
    print(f"  (excluding first {config.TREND_LOOKBACK_MINUTES} min of trajectory from crossing "
          f"search — health_state has a Kalman warm-up transient there, not a real early detection.)")

    def first_crossing_time(df_, cond, warmup_minutes=config.TREND_LOOKBACK_MINUTES):
        warm_cutoff = df_[config.COL_TIMESTAMP].min() + pd.Timedelta(minutes=warmup_minutes)
        hits = df_[cond & (df_[config.COL_TIMESTAMP] >= warm_cutoff)]
        return hits[config.COL_TIMESTAMP].min() if len(hits) else None

    sensor_warn_t = first_crossing_time(
        raw, (raw[config.COL_VIBRATION] >= warn_onset[config.COL_VIBRATION])
        | (raw[config.COL_TEMPERATURE] >= warn_onset[config.COL_TEMPERATURE])
        | (raw[config.COL_CURRENT] >= warn_onset[config.COL_CURRENT])
    )
    sensor_crit_t = first_crossing_time(
        raw, (raw[config.COL_VIBRATION] >= crit_onset[config.COL_VIBRATION])
        | (raw[config.COL_TEMPERATURE] >= crit_onset[config.COL_TEMPERATURE])
        | (raw[config.COL_CURRENT] >= crit_onset[config.COL_CURRENT])
    )
    health_warn_t = first_crossing_time(merged, merged["health_state"] <= config.MAINTENANCE_HEALTH_INSPECT)
    health_crit_t = first_crossing_time(merged, merged["health_state"] <= config.FAILURE_HEALTH_THRESHOLD)

    lags = {}

    def report(label, key, sensor_t, health_t):
        if sensor_t is None or health_t is None:
            print(f"  {label}: could not evaluate (no crossing found in one or both signals)")
            return
        lag = (health_t - sensor_t).total_seconds() / 60
        direction = "health lags sensor by" if lag > 0 else "health leads sensor by"
        print(f"  {label}: sensor onset {sensor_t}, health-threshold crossing {health_t} "
              f"-> {direction} {abs(lag):.0f} min")
        lags[key] = lag

    report(f"Warning (MAINTENANCE_HEALTH_INSPECT={config.MAINTENANCE_HEALTH_INSPECT})",
           "warn", sensor_warn_t, health_warn_t)
    report(f"Critical (FAILURE_HEALTH_THRESHOLD={config.FAILURE_HEALTH_THRESHOLD})",
           "crit", sensor_crit_t, health_crit_t)
    print(
        "\n  Read this as a rough cross-check, not a verdict: a large lag means the\n"
        "  health-based threshold fires later than sensor readings alone already\n"
        "  suggest trouble; a large lead means the opposite. Small lags are expected\n"
        "  — health_state is Kalman-smoothed by design.\n"
    )

    if lags:
        worst_lag_days = max(abs(v) for v in lags.values()) / (24 * 60)
        if worst_lag_days <= 1.0:
            findings.note("CONSISTENCY", "Threshold sanity check (informational)",
                          f"health-based thresholds cross within {worst_lag_days:.1f} days of the empirical sensor onset (weak, label-derived reference — see docstring).")
        else:
            findings.note("CONSISTENCY", "Threshold sanity check (informational)",
                          f"health-based thresholds cross {worst_lag_days:.1f} days from the empirical sensor onset — worth a look, but this reference is label-derived, not an independent spec (see docstring).")
    else:
        findings.note("CONSISTENCY", "Threshold sanity check (informational)",
                      "could not evaluate — no clean crossing found in one or both signals.")

    return {"warn_onset": warn_onset, "crit_onset": crit_onset,
            "sensor_warn_t": sensor_warn_t, "sensor_crit_t": sensor_crit_t,
            "health_warn_t": health_warn_t, "health_crit_t": health_crit_t}


# ===========================================================================
# PLOTTING — one dashboard: verdict banner, per-dimension findings cards,
# then the supporting evidence panels grouped by dimension.
# ===========================================================================

def _draw_card(ax, title, lines):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((0.02, 0.02), 0.96, 0.96, transform=ax.transAxes,
                                facecolor=COLOR["card_bg"], edgecolor=COLOR["grid"], linewidth=1.4, zorder=1))
    ax.add_patch(plt.Rectangle((0.02, 0.88), 0.96, 0.10, transform=ax.transAxes,
                                facecolor=COLOR["primary"], edgecolor="none", zorder=2))
    ax.text(0.5, 0.93, title, transform=ax.transAxes, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color="white", zorder=3)
    body = "\n".join(lines)
    ax.text(0.06, 0.80, body, transform=ax.transAxes, ha="left", va="top",
            fontsize=7.3, color=COLOR["ink"], zorder=3, linespacing=1.6)


def _wrap(text, width=58):
    import textwrap
    return "\n".join(textwrap.wrap(text, width=width))


def plot_report(findings: Findings, model_trained_at: str,
                 train_scores, holdout_scores, merged, y_true,
                 auc_raw, auc_state, calibration_results,
                 perturbation_results, refwindow_results,
                 batch_rt_result, repro_result):
    fig = plt.figure(figsize=(24, 30))
    gs = gridspec.GridSpec(
        6, 4, figure=fig,
        height_ratios=[0.55, 1.35, 1.35, 1.35, 1.35, 1.35],
        hspace=0.68, wspace=0.34,
        left=0.075, right=0.975, top=0.975, bottom=0.03,
    )

    # --- Row 0: title banner (no verdict) -----------------------------------
    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.axis("off")
    ax_banner.add_patch(plt.Rectangle((0, 0), 1, 1, transform=ax_banner.transAxes,
                                       facecolor=COLOR["card_bg"], edgecolor=COLOR["grid"], linewidth=1.2))
    ax_banner.text(0.015, 0.68, "SPINDLE CONDITION MONITORING — RELIABILITY TEST RESULTS",
                    transform=ax_banner.transAxes, ha="left", va="center",
                    fontsize=16, fontweight="bold", color=COLOR["ink"])
    ax_banner.text(0.015, 0.28,
                    "12 checks below, each reported on its own terms — no combined score. Read each finding's numbers and caveat directly.",
                    transform=ax_banner.transAxes, ha="left", va="center", fontsize=9.5, color=COLOR["muted"], style="italic")
    ax_banner.text(0.985, 0.68,
                    f"model trained: {model_trained_at}   |   report generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}",
                    transform=ax_banner.transAxes, ha="right", va="center", fontsize=9.5, color=COLOR["muted"])
    ax_banner.text(0.985, 0.30,
                    "Scope: one ~6.9-day trajectory, one labeled escalation event — these are findings against that data, not proof of generalization.",
                    transform=ax_banner.transAxes, ha="right", va="center", fontsize=9, color=COLOR["muted"], style="italic")

    # --- Row 1: four dimension cards, findings text only, no status ---------
    for col, dim in enumerate(DIMENSIONS):
        ax = fig.add_subplot(gs[1, col])
        entries = findings.dimension_entries(dim)
        max_lines = 4
        lines = []
        for e in entries[:max_lines]:
            lines.append(_wrap(f"{e['name']}: {e['detail']}", 46))
            lines.append("")
        if len(entries) > max_lines:
            lines.append(f"…and {len(entries) - max_lines} more (see console log).")
        _draw_card(ax, dim, lines)

    # --- Row 2: accuracy evidence -------------------------------------------

    ax = fig.add_subplot(gs[2, 0])
    ax.hist(train_scores, bins=30, alpha=0.65, label="train-half", color=COLOR["primary"])
    ax.hist(holdout_scores, bins=30, alpha=0.65, label="holdout-half", color=COLOR["warn"])
    ax.set_title("Overfitting check\ntrain vs. held-out score distributions")
    ax.set_xlabel("Isolation Forest score")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[2, 1])
    fpr_r, tpr_r, _ = roc_curve(y_true, -merged["health_raw"].values)
    fpr_s, tpr_s, _ = roc_curve(y_true, -merged["health_state"].values)
    ax.plot(fpr_r, tpr_r, label=f"health_raw (AUC={auc_raw:.3f})", alpha=0.75, color=COLOR["secondary"])
    ax.plot(fpr_s, tpr_s, label=f"health_state (AUC={auc_state:.3f})", linewidth=2.2, color=COLOR["primary"])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend(fontsize=8, frameon=False)

    ax = fig.add_subplot(gs[2, 2])
    prec, rec, _ = precision_recall_curve(y_true, -merged["health_state"].values)
    ax.plot(rec, prec, linewidth=2.2, color=COLOR["normal"])
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve\n(health_state)")

    ax = fig.add_subplot(gs[2, 3])
    ax.hist(merged.loc[y_true == 0, "health_state"], bins=30, alpha=0.65, label="normal", color=COLOR["normal"])
    ax.hist(merged.loc[y_true == 1, "health_state"], bins=30, alpha=0.65, label="warning/critical", color=COLOR["bad"])
    ax.set_xlabel("health_state (%)")
    ax.set_title("Health-state distribution\nby true label")
    ax.legend(fontsize=8, frameon=False)

    # --- Row 3: health over time + confusion matrices -----------------------
    ax = fig.add_subplot(gs[3, 0:2])
    colors = np.where(y_true == 1, COLOR["bad"], COLOR["normal"])
    ax.scatter(merged[config.COL_TIMESTAMP], merged["health_state"], c=colors, s=3, alpha=0.55, linewidths=0)
    ax.axhline(config.FAILURE_HEALTH_THRESHOLD, color=COLOR["ink"], linestyle="--", alpha=0.5,
               label=f"FAILURE_HEALTH_THRESHOLD={config.FAILURE_HEALTH_THRESHOLD}")
    ax.set_title("health_state over time (red=labeled bad, green=labeled normal)")
    ax.set_ylabel("health_state (%)")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(fontsize=8, frameon=False)

    def _cm_panel(ax, pred, y_true_, cmap, title, labels):
        cm = np.array([
            [((pred == 0) & (y_true_ == 0)).sum(), ((pred == 1) & (y_true_ == 0)).sum()],
            [((pred == 0) & (y_true_ == 1)).sum(), ((pred == 1) & (y_true_ == 1)).sum()],
        ])
        ax.imshow(cm, cmap=cmap)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=13, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(labels[0])
        ax.set_yticks([0, 1]); ax.set_yticklabels(labels[1])
        ax.set_title(title)

    ax = fig.add_subplot(gs[3, 2])
    pred_threshold = (merged["health_state"] <= config.FAILURE_HEALTH_THRESHOLD).astype(int).values
    _cm_panel(ax, pred_threshold, y_true, "Blues",
              f"Confusion matrix\nhealth_state <= {config.FAILURE_HEALTH_THRESHOLD}",
              (["Pred Fine", "Pred Bad"], ["Actual Fine", "Actual Bad"]))

    ax = fig.add_subplot(gs[3, 3])
    pred_maint = (merged["maintenance_level"] == "CRITICAL").astype(int).values
    _cm_panel(ax, pred_maint, y_true, "Oranges",
              "Confusion matrix\nmaintenance_level == CRITICAL",
              (["Pred Fine", "Pred CRITICAL"], ["Actual Fine", "Actual Bad"]))

    # --- Row 4: calibration + stability --------------------------------------
    ax = fig.add_subplot(gs[4, 0])
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    palette_cycle = [COLOR["primary"], COLOR["secondary"], COLOR["warn"], COLOR["bad"], COLOR["normal"], COLOR["muted"]]
    for i, (h, res) in enumerate(calibration_results.items()):
        ax.plot(res["bin_pred"], res["bin_actual"], marker="o", color=palette_cycle[i % len(palette_cycle)],
                label=f"{h}d (Brier={res['brier']:.3f})")
    ax.set_xlabel("Predicted failure probability"); ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=7, frameon=False)

    ax = fig.add_subplot(gs[4, 1])
    if calibration_results:
        hs = list(calibration_results.keys())
        briers = [calibration_results[h]["brier"] for h in hs]
        bar_colors = [COLOR["primary"] if h in config.FAILURE_PROB_HORIZONS_DAYS else COLOR["muted"] for h in hs]
        bars = ax.bar([str(h) + "d" for h in hs], briers, color=bar_colors)
        for h, res in calibration_results.items():
            if res.get("ci"):
                pass
        ax.axhline(0.25, color=COLOR["bad"], linestyle="--", alpha=0.6, label="uninformative (constant 0.5)")
        ax.set_ylabel("Brier score (lower is better)")
        ax.set_title("Brier score by horizon\n(blue=operational, grey=extended/reference)")
        ax.legend(fontsize=8, frameon=False)
    else:
        ax.text(0.5, 0.5, "No horizons had\nenough evaluable rows", ha="center", va="center")
        ax.set_title("Brier score by horizon")

    ax = fig.add_subplot(gs[4, 2])
    if perturbation_results:
        levels = sorted(perturbation_results.keys())
        corrs = [perturbation_results[l]["corr_min"] for l in levels]
        flips = [perturbation_results[l]["flip_rate_mean"] * 100 for l in levels]
        ax2 = ax.twinx()
        ax.plot([l * 100 for l in levels], corrs, "o-", color=COLOR["primary"], label="health_state corr (worst rep)")
        ax2.plot([l * 100 for l in levels], flips, "s--", color=COLOR["bad"], label="maintenance_level flip rate")
        ax.set_ylim(min(0.9, min(corrs) - 0.02), 1.005)
        ax.set_xlabel("Feature noise (% of feature std)")
        ax.set_ylabel("health_state correlation", color=COLOR["primary"])
        ax2.set_ylabel("flip rate (%)", color=COLOR["bad"])
        ax.set_title("Perturbation robustness")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, frameon=False, loc="lower left")
    else:
        ax.text(0.5, 0.5, "No perturbation\nresults", ha="center", va="center")
        ax.set_title("Perturbation robustness")

    ax = fig.add_subplot(gs[4, 3])
    if refwindow_results:
        shifts = sorted(refwindow_results.keys())
        corrs = [refwindow_results[s]["corr"] for s in shifts]
        bar_colors = [COLOR["pass"] if c >= 0.98 else (COLOR["warn"] if c >= 0.90 else COLOR["fail"]) for c in corrs]
        ax.bar([f"{s:+d}m" for s in shifts], corrs, color=bar_colors)
        ax.axhline(0.98, color=COLOR["muted"], linestyle="--", alpha=0.6)
        ax.set_ylim(min(0.8, min(corrs) - 0.02), 1.01)
        ax.set_ylabel("health_raw correlation vs. shipped model")
        ax.set_title("Reference-window sensitivity\n(burn-in window shifted ±minutes)")
    else:
        ax.text(0.5, 0.5, "No reference-window\nresults", ha="center", va="center")
        ax.set_title("Reference-window sensitivity")

    # --- Row 5: consistency ---------------------------------------------------
    ax = fig.add_subplot(gs[5, 0:2])
    if batch_rt_result and batch_rt_result.get("n_compared", 0) > 0:
        cats = ["max Δ health_state (pp)", "maint. disagreement (%)"]
        vals = [batch_rt_result["max_diff_state"], (1 - batch_rt_result["maint_match"]) * 100]
        bar_colors = [COLOR["pass"] if vals[0] < 0.5 else (COLOR["warn"] if vals[0] < 2 else COLOR["fail"]),
                      COLOR["pass"] if vals[1] == 0 else (COLOR["warn"] if vals[1] < 2 else COLOR["fail"])]
        ax.barh(cats, vals, color=bar_colors, height=0.5)
        ax.set_xlim(0, max(vals) * 1.35 + 0.05)
        for i, v in enumerate(vals):
            ax.text(v + max(vals) * 0.03, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_title(f"Batch vs. real-time equivalence\n"
                     f"n={batch_rt_result['n_compared']} tick-aligned readings compared, both lower-is-better", fontsize=10.5)
        ax.set_xlabel("value (0 = perfect agreement)")
        ax.tick_params(axis="y", labelsize=9)
        fig.canvas.draw()
    else:
        ax.text(0.5, 0.5, "No batch-vs-real-time\nresult", ha="center", va="center")
        ax.set_title("Batch vs. real-time equivalence")

    ax = fig.add_subplot(gs[5, 2:4])
    ax.axis("off")
    repro_status = "exact" if repro_result and repro_result.get("exact") else (
        f"max diff {repro_result['max_diff']:.2e}" if repro_result else "n/a")
    golden_entries = [e for e in findings.entries if e["name"].startswith("Cross-run regression")]
    golden_line = golden_entries[-1]["detail"] if golden_entries else "n/a"
    ax.add_patch(plt.Rectangle((0.02, 0.05), 0.96, 0.9, transform=ax.transAxes,
                                facecolor=COLOR["card_bg"], edgecolor=COLOR["grid"], linewidth=1.2))
    ax.text(0.5, 0.85, "Reproducibility & cross-run regression", transform=ax.transAxes,
            ha="center", va="center", fontsize=11.5, fontweight="bold")
    ax.text(0.08, 0.60, f"Training reproducibility:  {repro_status}", transform=ax.transAxes,
            ha="left", va="center", fontsize=9.5)
    ax.text(0.08, 0.30, _wrap("Golden snapshot: " + golden_line, 70), transform=ax.transAxes,
            ha="left", va="top", fontsize=9.5)

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(REPORT_PATH, dpi=115)
    plt.close(fig)
    print(f"Saved graphical report -> {REPORT_PATH}")


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    findings = Findings()

    print("Loading model and features...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scorer, feature_cols, metadata = artifact_utils.load_artifacts()
    df = pd.read_csv(config.FEATURES_DATA_PATH, parse_dates=[config.COL_TIMESTAMP])
    print(f"Model trained: {metadata['trained_at']}\n")

    reference_timestamps = artifact_utils.load_reference_timestamps()
    if reference_timestamps is None:
        print(
            "  WARNING: no reference_timestamps.json in artifacts/ — this model was\n"
            "  saved before the actual reference set was tracked. Falling back to\n"
            "  re-deriving config.REFERENCE_WINDOW_MINUTES's naive first-N-rows window,\n"
            "  which is only correct if that's genuinely how this model was trained.\n"
            "  Retrain with the current train_isolation_forest.py to fix this properly.\n"
        )
        reference_df, _ = preprocessing.split_reference_window(df)
    else:
        reference_df = df[df[config.COL_TIMESTAMP].isin(reference_timestamps)].reset_index(drop=True)
        if len(reference_df) != len(reference_timestamps):
            print(
                f"  WARNING: {len(reference_timestamps)} reference timestamps saved, but "
                f"only {len(reference_df)} matched rows in {config.FEATURES_DATA_PATH}. "
                f"features.csv may be stale relative to the trained model — consider "
                f"deleting it and rerunning train_isolation_forest.py.\n"
            )

    # ---- ACCURACY -----------------------------------------------------------
    train_scores, holdout_scores = check_overfitting(reference_df, feature_cols, findings)

    print("Running full backtest (vectorized scoring + sequential Kalman/trend/maintenance)...")
    backtest_df = run_backtest(df, feature_cols, scorer)
    print(f"Backtest complete: {len(backtest_df)} rows.\n")

    raw_labels = pd.read_csv(config.RAW_DATA_PATH)[[config.COL_TIMESTAMP, "health_status"]]
    raw_labels[config.COL_TIMESTAMP] = pd.to_datetime(raw_labels[config.COL_TIMESTAMP])
    raw_labels = raw_labels.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)
    raw_labels["pos"] = np.arange(len(raw_labels))
    raw_labels["is_bad"] = (raw_labels["health_status"] != "normal").astype(int)
    is_bad_array = raw_labels["is_bad"].values

    merged = backtest_df.merge(raw_labels[[config.COL_TIMESTAMP, "health_status", "pos"]],
                                on=config.COL_TIMESTAMP, how="left")
    y_true = (merged["health_status"] != "normal").astype(int).values

    # config.RAW_DATA_PATH may legitimately have only one health_status
    # class (it does right now: spindle_train.csv is 100% 'normal', by
    # design — see config.py). AUC/confusion matrices need both classes
    # to mean anything; when the training file can't provide that, fall
    # back to config.PREDICT_DATA_PATH (a separate trajectory, never seen
    # by the reference set) for every label-dependent check below. This
    # keeps check_discrimination()/check_calibration()/
    # check_threshold_calibration() completely unchanged — they just
    # receive different (merged, y_true, is_bad_array) depending on which
    # source actually has something to measure against.
    train_labels_usable = 0 < y_true.sum() < len(y_true)
    if train_labels_usable:
        print(f"Label source for accuracy/calibration checks: {config.RAW_DATA_PATH} "
              f"(both classes present: {y_true.sum()} bad / {(y_true == 0).sum()} normal).\n")
    else:
        only_class = "normal" if y_true.sum() == 0 else "bad"
        print(
            f"NOTE: {config.RAW_DATA_PATH} is entirely '{only_class}' "
            f"({len(y_true)} rows, 0 of the other class) — AUC and confusion matrices\n"
            f"against it would be undefined/meaningless, not just weak. Falling back to\n"
            f"{config.PREDICT_DATA_PATH} (a separate trajectory the reference set never saw)\n"
            f"for every check below that needs both classes.\n"
        )
        holdout = build_holdout_merged(scorer, feature_cols)
        if holdout is None:
            raise RuntimeError(
                f"{config.RAW_DATA_PATH} has only '{only_class}' rows AND "
                f"{config.PREDICT_DATA_PATH} has no health_status column — no data source "
                f"available for any accuracy/calibration check. Provide a labeled, "
                f"mixed-class file for at least one of these paths."
            )
        # NOTE: deliberately NOT reassigning backtest_df/df here — those
        # stay the training-file versions for the label-independent
        # STABILITY/CONSISTENCY checks below (perturbation robustness,
        # fault injection, reference-window sensitivity, reproducibility,
        # batch-vs-realtime, golden snapshot), which pair df+backtest_df
        # together and don't use labels at all. Only the label-dependent
        # (merged, y_true, is_bad_array) swap to the held-out file.
        merged, y_true, is_bad_array, _holdout_backtest_df = holdout
        findings.note(
            "ACCURACY", "Evaluation data source",
            f"{config.RAW_DATA_PATH} has no '{only_class == 'normal' and 'bad' or 'normal'}' rows to "
            f"validate against — every check below ran on {config.PREDICT_DATA_PATH} instead "
            f"({y_true.sum()} bad / {(y_true == 0).sum()} normal, n={len(y_true)})."
        )

    auc_raw, auc_state, ap_state, conf_threshold, conf_maint, ci_raw, ci_state = check_discrimination(
        merged, y_true, findings)

    # ---- CALIBRATION ---------------------------------------------------------
    calibration_results = check_calibration(merged, is_bad_array, findings)

    # ---- STABILITY -------------------------------------------------------------
    perturbation_results = check_perturbation_robustness(df, feature_cols, scorer, backtest_df, findings)
    check_fault_injection(df, feature_cols, scorer, findings)
    raw_scores_full = scorer.score(df[feature_cols])
    health_raw_full = scorer.health_from_score(raw_scores_full)
    refwindow_results = check_reference_window_sensitivity(df, feature_cols, health_raw_full, reference_df, findings)

    # ---- CONSISTENCY -----------------------------------------------------------
    repro_result = check_reproducibility(reference_df, feature_cols, findings)
    batch_rt_result = check_batch_vs_realtime(feature_cols, findings)
    check_golden_snapshot(backtest_df, findings)
    check_threshold_calibration(merged, findings,
                                 raw_data_path=None if train_labels_usable else config.PREDICT_DATA_PATH)

    # ---- REPORT ---------------------------------------------------------------
    print()
    print("=" * 70)
    print(f"Done — {len(findings.entries)} findings recorded above (see each section for detail).")
    print("=" * 70)

    print("\nGenerating graphical report...")
    plot_report(findings, metadata["trained_at"], train_scores, holdout_scores, merged, y_true,
                auc_raw, auc_state, calibration_results, perturbation_results, refwindow_results,
                batch_rt_result, repro_result)


if __name__ == "__main__":
    main()
