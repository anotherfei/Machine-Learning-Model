# Production Evaluation Report

## Dataset

- Source: `data\raw\spindle_given.csv`
- Raw samples: 10,000
- Production predictions evaluated: 9,977
- Warm-up rows with no production output: 23
- Labeled degraded rows among evaluated predictions (warning/critical): 6,189
- Rows with a ground-truth critical state now/within 1 day(s): 6,451

Every evaluated prediction above was returned directly by `predict_realtime.SpindleMonitor.update()`.
The evaluator contains no duplicate anomaly, Kalman, trend, probability, RUL, or maintenance inference path.

## Health Estimation

Binary ground truth: `health_status != normal`. Predicted degraded: `health_state <= 75`.
Continuous risk score for ROC/PR: `(100 - health_state) / 100`.

- ROC-AUC: 0.9956
- PR-AUC: 0.9972
- Accuracy: 67.35%
- Precision: 65.52%
- Recall: 100.00%
- F1-score: 79.17%

Confusion Matrix:

| Actual \ Predicted | Normal | Degraded |
|---|---:|---:|
| Normal | 531 | 3,257 |
| Degraded | 0 | 6,189 |

## Failure Probability

Ground truth: a `critical` label occurs now or within the configured 1-day maintenance horizon.
Production probability field: `failure_probability_1d`.

- ROC-AUC: 0.9508
- PR-AUC: 0.9259
- Brier Score: 0.1629
- Calibration Error (10-bin ECE): 0.3310

## Remaining Useful Life

SKIPPED — the configured evaluation dataset does not contain an independent true-RUL label/column. The production `remaining_days` output is collected, but MAE/RMSE are not fabricated from health-status labels.

## Maintenance Recommendation

Predicted positive: production maintenance level is `WARN` or `CRITICAL`. Actual positive: a ground-truth `critical` state occurs now or within 1 day(s).

- True Positives: 6,451
- False Positives: 3,297
- True Negatives: 229
- False Negatives: 0
- Precision: 66.18%
- Recall: 100.00%
- False Alarm Rate: 93.51%
- Miss Rate: 0.00%

## Overall Result

EVALUATION COMPLETED

No model-quality PASS/FAIL thresholds are defined by the production-evaluation architecture, so this report does not invent an acceptance gate.

The production pipeline was evaluated directly using `predict_realtime.SpindleMonitor`. All reported metrics are computed from outputs returned by the same inference implementation used for deployment.
