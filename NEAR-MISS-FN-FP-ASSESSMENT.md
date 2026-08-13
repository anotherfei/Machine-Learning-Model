# Assessment: "Near Miss = suspected FN, Alert Review = suspected FP"

This is a short writeup of what was pasted in from another conversation (Near Miss / Alert
Review acting as false-negative / false-positive candidate queues, gated by "ML confidence
rate"), whether it's possible, and what was actually changed in this pass versus left alone.

## Is it possible?
Yes, mechanically. Nothing about the schema or the pipeline blocks it.

## Is "ML confidence rate" the right way to define it?
No, and the pasted conversation already gets this right — worth restating plainly since it's
the one part of the proposal that's a hard requirement, not a design choice: `isolation_forest.py`
is unsupervised. `anomaly_score`, `health_state`, and `failure_probability` are risk/condition
signals computed from the model's own reference distribution — none of them are a calibrated
probability that a specific prediction is wrong. Whether a given WARN was a false positive, or a
given OK was a false negative, is only knowable from a ground-truth outcome (a human or a
downstream event), never from the model's own score. Any implementation of this idea has to key
off human review outcomes, not a score threshold.

## What the codebase already does
More of this than the pasted conversation gives it credit for:
- **FP side is already exactly this.** `alerts.status='confirmed_normal'` *is* "a human confirmed
  this alert was a false positive." `retrain_service._candidate_rows()` already pulls only
  `confirmed_normal` alerts into `reference_candidates`, and `run_shadow_retrain()`'s Gate 1
  checks that the shadow model doesn't regress on that confirmed-normal set. This part of the
  proposal is already built, not hypothetical.
- **An FN feedback mechanism already exists**, just not connected to Near Miss: `regression_tests`
  (`POST /api/regression-tests`, admin-only) stores a description + time window + a minimum
  anomaly-risk floor, and Gate 2 of `run_shadow_retrain()` checks that a shadow model still scores
  at least that much risk in that window before it's allowed to ship. That is a false-negative
  regression check — it exists specifically so a retrain can't quietly stop catching something a
  human already flagged as concerning.

The real gap the pasted conversation identifies — "the confirmed-FN path does not exist yet, and
[Near Miss] does not feed those records into a training/evaluation dataset" — is accurate. Near
Miss review outcomes (`acknowledged`/`flagged`) went nowhere before this pass; flagging something
had no effect on retraining at all.

## What was implemented
The narrow, low-risk piece: **flagging a near-miss now creates a `regression_tests` row**
(`POST /api/near-miss/{id}/review` with `decision=flagged`, in both `api/main.py` and
`api/mock_main.py`). The window is ±`NEAR_MISS_REGRESSION_WINDOW_HOURS` (new runtime-config key,
default 1 hour) around the flagged tick, and the risk floor is the record's own `anomaly_score` at
flag time (clamped to `[0.05, 0.95]`) — i.e. "a future model must not become *less* sensitive than
the current model already was to this case." This reuses the exact machinery Gate 2 already runs on
every retrain, so a flagged near-miss now has a real, enforced effect on what's allowed to ship,
which is the thing that was actually missing.

## What was **not** implemented, and why
The larger restructuring — renaming Near Miss's review outcomes to ground-truth labels
(`confirmed_normal` / `confirmed_fn` instead of `acknowledged`/`flagged`), redefining Near Miss
eligibility beyond "OK + negative anomaly-score slope" (the proposal's own suggestion: add
failure-probability, health-trend, and persistence-count conditions), and building `confirmed_fn`
into its own reference-style table — was deliberately left alone:
- It changes an existing status vocabulary (`near_miss_reviews.status`) and a checked constraint
  that a test in this repo (`test_near_miss_has_own_review_workflow`) asserts on directly, plus the
  `acknowledged`/`flagged` action buttons in the UI. Renaming that is a real migration on a table
  that may already have rows in it, not just a code change.
- The eligibility-criteria rework (multi-condition Near Miss instead of slope-only) is a modeling
  decision — what actually predicts a missed detection — that needs to be validated against real
  data, not guessed at from inside a UI/backend pass. Getting it wrong makes Near Miss noisier, not
  better.
- Both of those are meaningfully riskier changes to a pipeline with retraining gates and promotion
  logic attached, in an environment where I was asked not to run anything (Python tests, TSX
  typecheck, or the retraining/validation code itself). The regression-tests wiring above is small
  enough to review by reading; the rename/eligibility rework isn't something I'd want to ship
  unverified.

If you do want the full rename later, the regression-tests wiring above already gives you the
FN-side data path — the remaining work is purely UI/schema vocabulary, not new backend logic.

## Addendum: do `MAINTENANCE_TREND_DEBOUNCE_TICKS`/`MAINTENANCE_TREND_RECOVERY_TICKS` mean these aren't "true" FP/FN?

Short answer: no — the debounce/recovery window is already baked into the thing being judged,
so a human confirming or flagging it is still judging a real decision, not a raw noisy score.

Longer version, traced through the actual code:
- `maintenance.py`'s `MaintenanceDebouncer` requires `MAINTENANCE_TREND_DEBOUNCE_TICKS` (5)
  consecutive confirming ticks to escalate, and the longer `MAINTENANCE_TREND_RECOVERY_TICKS`
  (15) to de-escalate — deliberately asymmetric, since a false "all clear" is worse than a slow
  one. While a candidate level hasn't hit its required tick count yet, the debouncer holds and
  keeps *reporting the previous, already-confirmed level*.
- `worker.py` only ever persists that reported (post-debounce) `level` — to `spindle_predictions`,
  to `alerts`, and (via the same `maintenance_level` column) to Near Miss eligibility. The raw,
  still-climbing/clearing candidate level is never written anywhere a reviewer sees it.
- So by the time anything reaches Alert Review or Near Miss, the debouncing has *already happened*.
  A `confirmed_normal` alert isn't "the reviewer disagreed with a jittery raw score" — it's "the
  reviewer disagreed with the level the debouncer had already confirmed and the system actually
  acted on." Same for a flagged Near Miss: `maintenance_level='OK'` there is the debounced OK, not
  a single noisy tick.
- That makes it a real FP/FN of the *deployed decision* (what evaluate_production.py's
  `y_maintenance_pred` also scores against) — the debounce ticks changed how quickly that decision
  was reached and how long it persisted, not whether the human's judgment of it is valid.

Where the ticks *do* matter is interpretation, not validity: two `confirmed_normal` alerts can
still mean different things — a WARN that took the full 5 ticks to earn and turned out fine, versus
one on the edge of clearing that would've self-resolved in another tick or two if RECOVERY_TICKS
were shorter. Neither is "not a real FP", but the first says more about the underlying signal
(`MAINTENANCE_TREND_DEBOUNCE_TICKS`, a review-time decision) than about hysteresis, and the second
says more about the hold-down window itself (`MAINTENANCE_TREND_RECOVERY_TICKS`, a policy knob).
The `maintenance_reason` field already carries this context (e.g. "Holding at WARN — needs N more
confirming ticks") and is shown on both the Alert and History detail views, so a reviewer isn't
judging blind — but nothing currently separates "FP where debounce config was the dominant factor"
from "FP where the model was just wrong" in aggregate counts. That split wasn't built here (it's
a metrics/analysis question, not a bug), but it's the natural next step if the debounce ticks
specifically — as opposed to review outcomes in general — are what you want to evaluate.

