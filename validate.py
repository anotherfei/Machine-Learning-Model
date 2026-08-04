"""
External validation & reliability suite — NOT part of the pipeline, and
deliberately kept separate from it.

This is the ONLY file in this repo that reads health_status. Nothing in
config.py, preprocessing.py, feature_engineering.py, isolation_forest.py,
kalman.py, trend_forecast.py, failure_probability.py, maintenance.py, or
predict_realtime.py ever loads it — that's the whole point of the
unsupervised design (see README.md). This script exists purely so you can
sanity-check the trained model against health_status yourself, the same
way you'd manually compare against any other label you happen to have —
it never feeds back into training or inference.

Run this after every retrain:

    python validate.py

Tests included, grouped the way they're usually grouped in practice:

  DISCRIMINATION  — can the model tell bad from normal at all?
    1. Overfitting check (train/holdout split of the reference window)
    2. ROC curve + AUC (health_raw vs health_state, pre/post Kalman+trend)
    3. Precision-Recall curve + average precision
    4. Threshold sweep (TPR/FPR at several health cutoffs)
    5. Confusion matrices (raw threshold vs full maintenance output)

  CALIBRATION / RELIABILITY — when it gives a probability, is it right?
    6. Brier score per forecast horizon
    7. Reliability diagrams (predicted vs. observed frequency, binned)

  All of this is also rendered to validation_report.png.
"""

import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score,
)

import config
import artifact_utils
from kalman import HealthKalmanFilter
import trend_forecast
import failure_probability
import maintenance

REPORT_PATH = config.VALIDATION_REPORT_PATH

# Short horizons actually testable against this dataset's ~6.9-day span —
# kept in sync with config.FAILURE_PROB_HORIZONS_DAYS so every horizon the
# pipeline actually reports gets a calibration check. 2d/3d are also
# tested here (beyond what's operational) specifically to confirm they
# stay excluded — see config.py's comment for the measured Brier scores
# that got them removed from FAILURE_PROB_HORIZONS_DAYS in the first place.
CALIBRATION_HORIZONS_DAYS = sorted(set(config.FAILURE_PROB_HORIZONS_DAYS + [2, 3]))


# ===========================================================================
# DISCRIMINATION
# ===========================================================================

def check_overfitting(reference_df: pd.DataFrame, feature_cols: list):
    """
    reference_df: the ACTUAL rows the currently-saved model was trained
    on (loaded via artifact_utils.load_reference_timestamps() in main()),
    not a re-derived guess. Previously this recomputed df.iloc[:ref_cutoff]
    internally regardless of how the model was actually trained — silently
    correct only when the naive first-N-rows window happened to be true,
    silently checking the wrong rows against a spec-based or search-based
    model otherwise. Fixed to test what was actually shipped.
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

    if gap_in_std > 2:
        print(
            "\n  WARNING: >2 std gap between train and held-out 'normal' data.\n"
            "  Before concluding this is overfitting, check whether the reference\n"
            "  set itself is actually internally consistent — e.g. compare raw\n"
            "  sensor means between the two halves. Real drift or heterogeneity\n"
            "  already present within the 'normal' rows will produce this same\n"
            "  gap and is a calibration problem (reference-set selection), not\n"
            "  overfitting."
        )
        for col in config.RAW_SENSOR_COLS:
            mean_col = f"{col.split('_')[0]}_mean"
            if mean_col in ref.columns:
                print(f"    {mean_col}: train-half avg={train_half[mean_col].mean():.3f}, "
                      f"holdout-half avg={holdout_half[mean_col].mean():.3f}")
    print()
    return train_scores, holdout_scores


# ===========================================================================
# FULL BACKTEST — vectorized scoring, sequential Kalman/trend/maintenance
# ===========================================================================

def run_backtest(df: pd.DataFrame, feature_cols: list, scorer):
    raw_scores = scorer.score(df[feature_cols])
    health_raw = scorer.health_from_score(raw_scores)

    kalman = HealthKalmanFilter(initial_level=health_raw[0])
    health_states, maint_levels = [], []
    slopes, residuals = [], []
    trend_minutes, trend_health = [], []

    for i, h in enumerate(health_raw):
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

        slope, intercept, resid = fit
        slopes.append(slope)
        residuals.append(resid)
        remaining = trend_forecast.remaining_days(i, hs, slope)
        prob_table = failure_probability.failure_probability_table(hs, slope, resid)
        rec = maintenance.recommend(hs, remaining, prob_table)
        maint_levels.append(rec["level"])

    out = df[[config.COL_TIMESTAMP]].copy()
    out["health_raw"] = health_raw
    out["health_state"] = health_states
    out["maintenance_level"] = maint_levels
    out["slope_per_minute"] = slopes
    out["residual_std"] = residuals
    return out


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


def check_discrimination(merged: pd.DataFrame, y_true: np.ndarray):
    print("=" * 70)
    print("2 & 3. DISCRIMINATION — ROC-AUC, PR-AUC, threshold sweep")
    print("=" * 70)

    auc_raw = roc_auc_score(y_true, -merged["health_raw"].values)
    auc_state = roc_auc_score(y_true, -merged["health_state"].values)
    ap_state = average_precision_score(y_true, -merged["health_state"].values)
    print(f"ROC-AUC — health_raw (pre-Kalman):          {auc_raw:.4f}")
    print(f"ROC-AUC — health_state (post-Kalman+trend): {auc_state:.4f}")
    print(f"PR-AUC (average precision) — health_state:  {ap_state:.4f}")
    if auc_state > auc_raw:
        print("  -> Kalman+trend layer IMPROVES discrimination over the raw anomaly score.")
    else:
        print("  -> Kalman+trend layer does NOT improve on the raw anomaly score here.")
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
    print("4. CONFUSION MATRICES")
    print("=" * 70)
    print(f"Total rows: {len(merged)}  |  Actually bad: {y_true.sum()}  |  Actually normal: {(y_true==0).sum()}")
    print()

    pred_threshold = (merged["health_state"] <= config.FAILURE_HEALTH_THRESHOLD).astype(int).values
    conf_threshold = confusion(pred_threshold, y_true, f"health_state <= {config.FAILURE_HEALTH_THRESHOLD} (raw threshold only)")

    pred_maint = (merged["maintenance_level"] == "CRITICAL").astype(int).values
    conf_maint = confusion(pred_maint, y_true, "maintenance_level == CRITICAL (full pipeline output)")

    return auc_raw, auc_state, ap_state, conf_threshold, conf_maint


# ===========================================================================
# CALIBRATION / RELIABILITY
# ===========================================================================

def check_calibration(merged: pd.DataFrame, is_bad_array: np.ndarray):
    print("=" * 70)
    print("5 & 6. CALIBRATION — Brier score and reliability diagrams")
    print("=" * 70)
    print(
        "Checks whether failure_probability_table()'s numbers mean what they\n"
        "claim: among all times the model said '70% chance', did the bad state\n"
        "actually occur about 70% of the time? This is a genuinely different\n"
        "property from discrimination (AUC) — a model can rank cases correctly\n"
        "while still being poorly calibrated.\n"
    )
    print(
        f"CAVEAT: this dataset spans ~{len(is_bad_array)/1440:.1f} days with exactly ONE\n"
        f"escalation event. config.FAILURE_PROB_HORIZONS_DAYS "
        f"({config.FAILURE_PROB_HORIZONS_DAYS}) were restricted to <=1 day\n"
        f"specifically because longer horizons (2d/3d, tested below for reference)\n"
        f"measured Brier scores worse than the uninformative baseline. Even these\n"
        f"shorter horizons draw from overlapping windows within a single event —\n"
        f"treat this as a first look, not a robust calibration guarantee; that\n"
        f"needs a fleet with many independent failure histories.\n"
    )

    n_total = len(is_bad_array)
    results = {}

    for h in CALIBRATION_HORIZONS_DAYS:
        h_minutes = int(h * 24 * 60)
        preds, actuals = [], []

        for idx, row in merged.iterrows():
            if pd.isna(row["slope_per_minute"]):
                continue
            pos = row["pos"]
            window_end = pos + h_minutes
            if window_end >= n_total:
                continue  # not enough future data for this horizon — censored, skip

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
        print(f"Horizon {h}d: n={len(preds)}, Brier score={brier:.4f} "
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

        results[h] = {"preds": preds, "actuals": actuals, "brier": brier,
                      "bin_pred": bin_pred, "bin_actual": bin_actual, "bin_n": bin_n}

    for h in config.FAILURE_PROB_HORIZONS_DAYS:
        if h not in CALIBRATION_HORIZONS_DAYS:
            h_minutes = int(h * 24 * 60)
            evaluable = int((merged["pos"] + h_minutes < n_total).sum())
            print(f"Horizon {h}d (operational, not calibration-tested): "
                  f"only {evaluable} rows would have complete future data — skipped.")
    print()
    return results


# ===========================================================================
# PLOTTING
# ===========================================================================

def plot_report(train_scores, holdout_scores, merged, y_true,
                 auc_raw, auc_state, calibration_results):
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))

    # 1. Overfitting: score distributions
    ax = axes[0, 0]
    ax.hist(train_scores, bins=30, alpha=0.6, label="train-half", color="steelblue")
    ax.hist(holdout_scores, bins=30, alpha=0.6, label="holdout-half", color="darkorange")
    ax.set_title("Overfitting check: reference window\ntrain vs. held-out score distributions")
    ax.set_xlabel("Isolation Forest score")
    ax.legend()

    # 2. ROC curve
    ax = axes[0, 1]
    fpr_r, tpr_r, _ = roc_curve(y_true, -merged["health_raw"].values)
    fpr_s, tpr_s, _ = roc_curve(y_true, -merged["health_state"].values)
    ax.plot(fpr_r, tpr_r, label=f"health_raw (AUC={auc_raw:.3f})", alpha=0.7)
    ax.plot(fpr_s, tpr_s, label=f"health_state (AUC={auc_state:.3f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()

    # 3. Precision-Recall curve
    ax = axes[0, 2]
    prec, rec, _ = precision_recall_curve(y_true, -merged["health_state"].values)
    ax.plot(rec, prec, linewidth=2, color="darkgreen")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve (health_state)")

    # 4. health_state distribution by class
    ax = axes[1, 0]
    ax.hist(merged.loc[y_true == 0, "health_state"], bins=30, alpha=0.6, label="normal", color="seagreen")
    ax.hist(merged.loc[y_true == 1, "health_state"], bins=30, alpha=0.6, label="warning/critical", color="firebrick")
    ax.set_xlabel("health_state (%)")
    ax.set_title("Health-state distribution by true label")
    ax.legend()

    # 5. health_state over time, colored by label
    ax = axes[1, 1]
    colors = np.where(y_true == 1, "firebrick", "seagreen")
    ax.scatter(merged[config.COL_TIMESTAMP], merged["health_state"], c=colors, s=2, alpha=0.5)
    ax.axhline(config.FAILURE_HEALTH_THRESHOLD, color="black", linestyle="--", alpha=0.5,
               label=f"FAILURE_HEALTH_THRESHOLD={config.FAILURE_HEALTH_THRESHOLD}")
    ax.set_title("health_state over time (red=labeled bad, green=labeled normal)")
    ax.set_ylabel("health_state (%)")
    ax.tick_params(axis="x", rotation=30)
    ax.legend()

    # 6. Confusion matrix heatmap — raw threshold
    ax = axes[1, 2]
    pred_threshold = (merged["health_state"] <= config.FAILURE_HEALTH_THRESHOLD).astype(int).values
    cm1 = np.array([
        [((pred_threshold == 0) & (y_true == 0)).sum(), ((pred_threshold == 1) & (y_true == 0)).sum()],
        [((pred_threshold == 0) & (y_true == 1)).sum(), ((pred_threshold == 1) & (y_true == 1)).sum()],
    ])
    im = ax.imshow(cm1, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm1[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Fine", "Pred Bad"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Fine", "Actual Bad"])
    ax.set_title(f"Confusion matrix\nhealth_state <= {config.FAILURE_HEALTH_THRESHOLD}")

    # 7. Confusion matrix heatmap — full maintenance output
    ax = axes[2, 0]
    pred_maint = (merged["maintenance_level"] == "CRITICAL").astype(int).values
    cm2 = np.array([
        [((pred_maint == 0) & (y_true == 0)).sum(), ((pred_maint == 1) & (y_true == 0)).sum()],
        [((pred_maint == 0) & (y_true == 1)).sum(), ((pred_maint == 1) & (y_true == 1)).sum()],
    ])
    im = ax.imshow(cm2, cmap="Oranges")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm2[i, j]), ha="center", va="center", fontsize=14)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Fine", "Pred CRITICAL"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual Fine", "Actual Bad"])
    ax.set_title("Confusion matrix\nmaintenance_level == CRITICAL")

    # 8. Reliability diagram
    ax = axes[2, 1]
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4, label="perfect calibration")
    for h, res in calibration_results.items():
        ax.plot(res["bin_pred"], res["bin_actual"], marker="o", label=f"{h}d (Brier={res['brier']:.3f})")
    ax.set_xlabel("Predicted failure probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability diagram")
    ax.legend(fontsize=8)

    # 9. Brier scores bar chart
    ax = axes[2, 2]
    if calibration_results:
        hs = list(calibration_results.keys())
        briers = [calibration_results[h]["brier"] for h in hs]
        ax.bar([str(h) + "d" for h in hs], briers, color="slateblue")
        ax.axhline(0.25, color="red", linestyle="--", alpha=0.5, label="uninformative (constant 0.5)")
        ax.set_ylabel("Brier score (lower is better)")
        ax.set_title("Brier score by horizon")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No horizons had\nenough evaluable rows", ha="center", va="center")
        ax.set_title("Brier score by horizon")

    plt.tight_layout()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    plt.savefig(REPORT_PATH, dpi=120)
    print(f"Saved graphical report -> {REPORT_PATH}")


# ===========================================================================
# THRESHOLD SANITY CHECK — health-based vs sensor-based severity onset
# ===========================================================================

def check_threshold_calibration(merged: pd.DataFrame):
    """
    Sanity-checks config.FAILURE_HEALTH_THRESHOLD (20) and
    config.MAINTENANCE_HEALTH_INSPECT (40) — currently, per README.md,
    "starting points... tuned by inspecting behavior, not fit to ground
    truth" — against when raw sensor readings actually start looking like
    warning/critical rows in the labeled data.

    IMPORTANT: there is no manufacturer warning/critical spec for this
    machine anywhere in this repo or in Predictive_Maintenance.zip (that
    codebase's WARNING/CRITICAL labels come from KMeans clustering of
    autoencoder reconstruction error, not fixed sensor bounds — checked
    directly, no such thresholds exist there to borrow). The "onset"
    values below are instead derived empirically from health_status itself:
    the midpoint between each class's IQR edges, per sensor. This is a
    genuinely different, weaker kind of evidence than a real spec bound —
    it's "where the labels already say severity changes," not an
    independent physical reference — but it's still an out-of-band check
    against the health-percentage thresholds, since maintenance.py never
    sees health_status.

    This function only reads health_status (already validate.py's one
    allowed exception) and is purely diagnostic — it does not feed back
    into config.py or maintenance.py automatically.
    """
    print("=" * 70)
    print("8. THRESHOLD SANITY CHECK — health% thresholds vs. sensor onset")
    print("=" * 70)

    raw = pd.read_csv(config.RAW_DATA_PATH)
    raw[config.COL_TIMESTAMP] = pd.to_datetime(raw[config.COL_TIMESTAMP])
    raw = raw.sort_values(config.COL_TIMESTAMP).reset_index(drop=True)

    iqr = raw.groupby("health_status")[config.RAW_SENSOR_COLS].quantile([0.25, 0.75]).unstack()
    missing = [s for s in ("normal", "warning", "critical") if s not in iqr.index]
    if missing:
        print(f"  Skipped: health_status is missing classes {missing} in this dataset.\n")
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
          f"search — health_state has a Kalman warm-up transient there, confirmed separately, "
          f"that isn't a real early detection.)")

    def first_crossing_time(df, cond, warmup_minutes=config.TREND_LOOKBACK_MINUTES):
        """
        Excludes the first `warmup_minutes` of the trajectory. health_state
        starts at the Kalman filter's initial_level (= the very first raw
        score) and needs time to converge — confirmed separately that
        health_state can start near ~19% and climb to its stable ~90s over
        roughly the first few hundred minutes, regardless of true condition.
        Without this cutoff, that transient gets mistaken for an early
        critical crossing on every run, making the lead/lag number below
        meaningless. TREND_LOOKBACK_MINUTES is reused here only as a
        reasonable existing "the pipeline considers this its warm-up scale"
        constant, not because it's derived for this purpose specifically.
        """
        warm_cutoff = df[config.COL_TIMESTAMP].min() + pd.Timedelta(minutes=warmup_minutes)
        hits = df[cond & (df[config.COL_TIMESTAMP] >= warm_cutoff)]
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

    def report(label, sensor_t, health_t):
        if sensor_t is None or health_t is None:
            print(f"  {label}: could not evaluate (no crossing found in one or both signals)")
            return
        lag = (health_t - sensor_t).total_seconds() / 60
        direction = "health lags sensor by" if lag > 0 else "health leads sensor by"
        print(f"  {label}: sensor onset {sensor_t}, health-threshold crossing {health_t} "
              f"-> {direction} {abs(lag):.0f} min")

    report(f"Warning (MAINTENANCE_HEALTH_INSPECT={config.MAINTENANCE_HEALTH_INSPECT})",
           sensor_warn_t, health_warn_t)
    report(f"Critical (FAILURE_HEALTH_THRESHOLD={config.FAILURE_HEALTH_THRESHOLD})",
           sensor_crit_t, health_crit_t)
    print(
        "\n  Read this as a rough cross-check, not a verdict: a large lag means the\n"
        "  health-based threshold fires later than sensor readings alone already\n"
        "  suggest trouble, and could be tightened (raised); a large lead means\n"
        "  the opposite. Small lags are expected — health_state is Kalman-smoothed\n"
        "  by design and shouldn't react to a single instantaneous reading.\n"
    )
    return {"warn_onset": warn_onset, "crit_onset": crit_onset,
            "sensor_warn_t": sensor_warn_t, "sensor_crit_t": sensor_crit_t,
            "health_warn_t": health_warn_t, "health_crit_t": health_crit_t}


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    print("Loading model and features...")
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
        ref_cutoff = min(config.REFERENCE_WINDOW_MINUTES, len(df))
        reference_df = df.iloc[:ref_cutoff].reset_index(drop=True)
    else:
        reference_df = df[df[config.COL_TIMESTAMP].isin(reference_timestamps)].reset_index(drop=True)
        if len(reference_df) != len(reference_timestamps):
            print(
                f"  WARNING: {len(reference_timestamps)} reference timestamps saved, but "
                f"only {len(reference_df)} matched rows in {config.FEATURES_DATA_PATH}. "
                f"features.csv may be stale relative to the trained model — consider "
                f"deleting it and rerunning train_isolation_forest.py.\n"
            )

    train_scores, holdout_scores = check_overfitting(reference_df, feature_cols)

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

    auc_raw, auc_state, ap_state, conf_threshold, conf_maint = check_discrimination(merged, y_true)
    calibration_results = check_calibration(merged, is_bad_array)
    check_threshold_calibration(merged)

    print("Generating graphical report...")
    plot_report(train_scores, holdout_scores, merged, y_true, auc_raw, auc_state, calibration_results)


if __name__ == "__main__":
    main()
