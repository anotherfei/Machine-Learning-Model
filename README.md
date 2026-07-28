# Gearbox RUL — ML Template

Predicts Remaining Useful Life (RUL) for a gearbox from three sensors:
vibration, current, temperature. Tested end-to-end on a synthetic sample
dataset included in `data/raw/raw.csv` — replace it with your real data.

## Setup

```bash
pip install -r requirements.txt
```

## Your CSV format

Required columns (rename in `config.py` if your headers differ):

```
unit_id, timestamp, vibration, current, temperature
```

Optional: a `RUL` column with ground-truth remaining-life values. If you
don't have one, set `PIECEWISE_LABELING = True` in `config.py` and RUL will
be derived automatically as "rows remaining until the end of that unit's
trajectory," capped at `RUL_CAP`. This only makes sense if each unit's data
actually runs to failure — if your data is mid-life snapshots with no
failure endpoint, you need real RUL labels, not derived ones.

Put your file at `data/raw/raw.csv`.

## Run

```bash
# 1. Clean + label
python preprocessing.py

# 2. Engineer rolling features
python feature_engineering.py

# 3a. Train + save one model
python train.py --model lightgbm

# 3b. OR cross-validate and compare all models fairly (recommended)
python compare_models.py

# 4. Predict on new data with a saved model
python predict.py --model lightgbm --input path/to/new_data.csv
```

## Design decisions baked into this template (don't undo these casually)

- **Splitting is always by `unit_id`, never by row.** Random row splits leak
  adjacent time-window features between train/test and inflate every metric.
- **PHM08 asymmetric score is reported alongside RMSE/MAE/R2.** It penalizes
  optimistic (late) RUL predictions harder — the operationally dangerous
  failure mode. Model rankings can differ between RMSE and PHM08; PHM08 is
  the one that matters for deployment decisions.
- **Rolling windows with insufficient history produce NaN and get dropped,
  never imputed.** Manufacturing a "healthy-looking" value at trajectory
  start would corrupt the degradation signal.
- **`artifacts/metadata.json` stores a hash of the feature-engineering
  config used at train time.** `predict.py` recomputes it and refuses to
  run if `config.py` has drifted since training (e.g. someone changed
  `WINDOW_SIZE`) — this catches silent value corruption that a
  column-name check alone would miss.
- **Stacking (`models/stacking.py`) is excluded from `compare_models.py` by
  default.** Only wire it in after confirming you have enough independent
  unit trajectories (rule of thumb: 10+) — otherwise the meta-learner
  overfits. See the docstring in that file for the nested-CV approach if
  you do build it out.

## Project structure

```
config.py                 # all tunable constants — nothing else should hardcode these
preprocessing.py          # load, clean, group-aware split
feature_engineering.py    # rolling stats per unit (vibration/current/temperature)
metrics.py                # MAE, RMSE, R2, PHM08 — same function for every model
artifact_utils.py         # save/load model + scaler + feature list + config hash
models/
  base.py                 # shared interface
  linear_regression.py
  random_forest.py
  svr.py
  lightgbm_model.py
  stacking.py              # stub — see notes above before using
train.py                  # train ONE model, holdout split
compare_models.py         # train ALL models, GroupKFold CV, comparison table
predict.py                # inference with strict feature validation
artifacts/                # saved .pkl models, scaler, feature_columns.json, metadata.json
data/raw/, data/processed/
```

## Before you trust the results

- Check `n_features` vs. row count per unit — if you have very few units
  (single digits), CV folds will be noisy; treat scores as directional,
  not precise.
- Confirm your vibration sampling rate actually supports the rolling
  statistics used (RMS, kurtosis, crest factor need enough samples per
  window to be meaningful — a window of 30 points from a 1 Hz signal is
  very different from 30 points at 10 kHz).
- `WINDOW_SIZE` in `config.py` is in **rows**, not seconds — convert from
  your real sampling rate before trusting the defaults.
