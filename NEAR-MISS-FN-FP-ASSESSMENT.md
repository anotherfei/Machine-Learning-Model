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
