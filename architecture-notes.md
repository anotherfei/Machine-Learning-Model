# Architecture notes

## Initial model training

`train_isolation_forest.py` fits the Isolation Forest on a reference/
commissioning window. Where that window comes from is `config.REFERENCE_SOURCE`:

- **`"live"` (default)** — pulled directly from the same Postgres table
  and connection `worker.py`/`db.py` use in production, restricted to
  `[REFERENCE_WINDOW_START, REFERENCE_WINDOW_END)`. Training and
  production share one schema and one source of truth: no separate
  offline file whose columns can silently drift from what the live
  sensor actually emits. Those two timestamps must be set explicitly
  (`config.py`, or `--start`/`--end` on the command line) — the script
  refuses to guess a window from "whatever the table currently holds",
  since the table keeps growing and that would make every run pull a
  different, unreproducible reference set. Pin them once you've
  identified a confirmed-healthy commissioning stretch in the live data.
  Use `--all-machines` to discover all source machine IDs, or repeat
  `--machine-id` to select an explicit subset. Large ranges use two bounded
  streaming passes per machine. The first considers every clean source row
  while learning a bounded motion profile. The second preserves the previous
  `WINDOW_SIZE-1` clean rows across database chunks, so rolling features are
  identical to full-trajectory feature construction, then automatically gates
  invalid/non-running/spec-ineligible rows. Every eligible row receives a
  deterministic random priority; the bounded per-machine reservoir therefore
  remains representative without loading the source range into memory. The
  retained pools are balanced to the same count before they are combined,
  preventing the highest-volume machine from dominating the shared forest.
  The newest balanced 20% (at least 20 rows per machine) is reserved as a
  forward validation holdout before normalizer, tree, or condition-anchor
  fitting. `artifact_utils.save_artifacts()` records the fit and validation
  feature tables separately, plus machine counts and row identities, for
  auditability.
- **`"csv"`** — the original offline-file behavior, reading
  `config.RAW_DATA_PATH`. Kept for offline experimentation and for CI/test
  fixtures that shouldn't need a reachable database.

For live training, the automatic machine-local operating-state pass excludes
`STOPPED` and `STARTING` rows without removing them before rolling feature
construction. The configured `SPEC_MAX` bounds (or `--full` to skip that test)
then decide which confirmed-running rows are accepted as healthy. The user
does not manually filter individual readings. Only use `--full` when every
running portion of the window is independently confirmed healthy.
`--include-non-running` is an explicit manual override for a window
independently known to contain running data only.

Before fitting the shared tree, every engineered feature is normalized with
that machine's confirmed-healthy median and robust IQR/MAD scale. The fitted
normalizers are part of the model bundle and are mandatory at inference; a
new machine cannot be scored until commissioning training includes it. Raw
reference features remain stored in physical units so a shadow retrain can
fit its own proposed normalizers without compounding an older transform.

After the shared tree is fitted, the trainer creates a robust residual
condition-score anchor for each machine from its normalized fit-row score
distribution. These anchors are automatic and switch with the model. Manual
web/CLI recalibration is deliberately unsupported because selecting data from
the current model's own `OK` decisions creates a circular, drifting baseline.

## Shared-model retraining

The active bundle carries `reference_features.csv`: the exact balanced,
machine-aware feature corpus used to fit its tree, and
`validation_features.csv`: equal per-machine confirmed-normal evidence that
was excluded from fitting and calibration. Shadow retraining starts from these
artifacts instead of trying to recover pooled features from a timestamp-only
CSV lookup. Legacy bundles create this separation on their first upgraded
retrain; newly commissioned bundles have it from initial training.

Only alerts that an operator marks `confirmed_normal` become retraining
candidates. The scheduler evaluates the batch and age thresholds per machine.
Its evaluation interval and the regression-test window created from a flagged
near miss are runtime policy stored in PostgreSQL and editable by an admin.
Candidates are cosine-deduplicated only against candidates from the same
machine, then merged into that machine's reference. Rebalancing gives every
machine the same row count and prioritizes new confirmed-normal rows when old
rows must be displaced. Retraining never onboards a new machine from alert
samples because those are a biased slice of its operating distribution. Add a
new machine by rerunning balanced initial training with a confirmed-healthy
commissioning window.

Each shadow model must pass the confirmed-normal false-positive gate for every
machine independently. Permanent false-negative regression windows are rebuilt
from the matching machine's live raw rows, transformed with that machine's
proposed normalizer, and scored with its proposed automatic anchor. A shadow
is promotable only if every machine-level holdout gate and every regression test
passes. Promotion switches the shared tree, machine normalizers, and condition
anchors together.

The scheduler and manual web action enqueue `retrain_jobs`; training does not
run inside the HTTP request. A PostgreSQL advisory lock plus the active-job
index enforce one fleet training process at a time. A passed shadow stages the
candidate IDs in `model_version_candidates`. They remain pending until an
administrator promotes that exact shadow, when they are consumed in the same
database transaction as activation. Deleting the shadow releases them. Only
one shadow can await a decision, so multiple proposals cannot reserve disjoint
candidate sets. Automatic retries suppress an unchanged validation rejection;
unexpected failures use the configured cooldown. Candidate IDs, active model,
material policy, pipeline hash, retraining-protocol source hash, and active
regression tests form the attempt fingerprint, so genuinely new evidence or
deployed validation logic is not blocked by an older result.

## Production

`React/Vite -> FastAPI -> PostgreSQL` for control-plane requests and history.

The production worker runs separately and owns `SpindleMonitor`, feature engineering, anomaly scoring, health estimation, forecasting, and maintenance recommendation. Results are written to `spindle_predictions` and streamed to the UI.

## Temporary mock mode

`React/Vite -> FastAPI Demo/mock_main.py -> SQLite Demo/mock_demo.db`.

The production worker is not started. The mock API exposes the same frontend-facing endpoints and generates synthetic live ticks for multiple machine IDs so the web UI can be verified without a PostgreSQL server or trained artifacts.

## Local ports

- Frontend: `5173`
- FastAPI: `8000`
- PostgreSQL production default: `5432`

Database configuration is read from `.env`. Server command-line flags are not parsed as database overrides.
