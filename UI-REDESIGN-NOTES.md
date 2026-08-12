# Frontend redesign notes

The React frontend was rebuilt around the existing backend contracts, with one new backend contract added afterward: a promote action for near-miss records (see "Near Miss actions" below).

## Main changes
- Replaced the prototype card grid with a unified machine-status board and compact live sensor strip.
- Added collapsible navigation, top status bar, responsive layout, and redesigned login experience.
- Added interactive drill-down modals/popovers instead of adding more permanent containers.
- Dashboard: status evidence, model/runtime context, mock-mode explanation, sensor channel details, anomaly explanation, live sparklines.
- Alerts: filters, structured table, row inspection, raw sensor disclosure, ±3-hour context, review actions inside the detail modal.
- Models: retraining summary, structured model list, validation report detail, promotion action in the model modal.
- History / Near Miss: search and state filters, structured records table, click-to-inspect sensor/failure-probability/full-backend details, plus a dedicated review workflow on Near Miss records and downstream-outcome tracking on History rows (see below).
- Thresholds: readable policy labels with backend keys preserved, hot-reload action, collapsible policy explanation.
- Environment: primary DB fields first, advanced values collapsed, credential handling explanation.
- Added responsive behavior for tablet/mobile widths.

The frontend remains presentation-only: ML, anomaly scoring, maintenance decisions, threshold validation, retraining, and database operations stay in the Python backend/worker.

## Near Miss actions
The Near Miss page was originally read-only (search/filter/inspect, no way to act on a record). It now has its own review workflow, structured the same way Alert review works but scoped to near-miss records rather than reusing the `alerts` table:
- New table `near_miss_reviews` (one row per `spindle_predictions.id`, added in `db_schema.py`/mock schema) tracks a status of `pending` / `acknowledged` / `flagged` per record, independent of the real alert severity levels.
- `GET /api/near-miss` takes a `status` filter (default `pending`, mirroring how Alert review defaults to pending) and returns each record's `review_status`.
- New endpoint `POST /api/near-miss/{prediction_id}/review` (in both `api/main.py` and `api/mock_main.py`) takes `{"decision": "acknowledged" | "flagged"}` and upserts into `near_miss_reviews`.
- The frontend detail modal shows "Acknowledge" / "Flag for follow-up" buttons when a record is pending, matching the Alert review modal's confirm/dismiss pattern. The table shows a "Review state" column instead of the maintenance-level column used elsewhere, since every near-miss row is `OK` by definition.

An earlier version of this feature inserted near-miss records directly into the `alerts` table with a hardcoded `WARN` level and the `trend_probability` trigger, so it could reuse the Alert review workflow. That was reverted: `trend_probability` specifically means "the automated failure-probability model crossed a threshold" (see `maintenance.py`), and these records never crossed that threshold — that's why they're near-misses and not alerts. Reusing it made a manual human escalation look like an automatic model detection, and forced a healthy record to display a fabricated WARN severity. The dedicated `near_miss_reviews` table avoids misrepresenting severity or trigger provenance.

## History outcome tracking
Prediction history rows had no way to tell whether a tick went on to become an alert or a near-miss review, and if so what happened to it. `GET /api/history` now joins in that downstream state:
- `api/main.py` uses a `LEFT JOIN LATERAL` against `alerts` (matched by `tick_timestamp`+`model_version`, most recent row if more than one) plus a `LEFT JOIN` against `near_miss_reviews` (matched by `prediction_id`), exposing `alert_status`/`alert_level`/`alert_trigger`/`alert_reviewed_by`/`alert_reviewed_at` and `near_miss_status`/`near_miss_reviewed_by`/`near_miss_reviewed_at`.
- `api/mock_main.py` does the equivalent per-row in Python, since SQLite has no `LATERAL` support.
- The History table gets an "Outcome" column with a badge — `Alert · <status>` or `Near miss · <status>` — colored the same way review outcomes are colored elsewhere (pending = neutral, confirmed/flagged = warning-ish, confirmed normal/acknowledged = normal). Rows with neither show "—".
- The detail modal shows the full outcome (status, level/trigger for alerts, who reviewed it and when) when a row was picked up by either workflow, and says plainly "Not surfaced in Alert review or Near miss" when it wasn't.
- Note: any alert rows created by the earlier (reverted) promote-to-alert feature are still sitting in the `alerts` table from prior testing — those will now show up in History as `Alert · <status>` with a `Trend forecast` trigger even though they were manually created. That's pre-existing data, not something this change can clean up without a migration/backfill script, which wasn't run per your instruction.

## History pagination + merged Status Review (second pass)
Three follow-up changes, made together because the second two both needed the same pagination plumbing as the first:

**1. History is now paginated server-side, with a configurable page size.**
`GET /api/history` no longer just returns an array capped at `limit`; it takes `limit`+`offset`, an optional `level` filter (`OK`/`WARN`/`CRITICAL`, pushed server-side since it's a plain column), and returns `{items, total, limit, offset}` so the frontend can render real page controls instead of a single capped list. `GET /api/alerts` and `GET /api/near-miss` were changed the same way for consistency (see point 2) — same response shape, same `limit`/`offset` params. The near-miss query already ran a window function per request; getting a total now means running the same `WITH x AS (...)` CTE twice (once wrapped in `count(*)`, once with `ORDER BY ... LIMIT/OFFSET`) rather than trying to fold a running count into the window query itself.

The free-text search box on History still filters client-side, but now only across whatever page is currently loaded — it no longer silently searches a fixed 200-row window like before. The surface heading says "`N` of `M` on this page" whenever search text is present, so that scope is visible rather than assumed. The state (`OK`/`WARN`/`CRITICAL`) filter, by contrast, is a real server-side query and its counts are exact.

A new `Pagination` component (page/pageSize state via a small `usePagination` hook, `«‹ Page X of Y ›»` plus a page-size `Select`, sizes 10/25/50/100) is shared by History, Alert review, and Near miss — all three now paginate the same way.

**2. Alert review and Near miss are merged into one "Status review" page**, replacing the two separate nav entries. It's one `PageHeader` with a segmented `Alert review` / `Near miss` tab switch; each tab is its own component (`AlertPanel`, `NearMissPanel`) with its own filters, table, and detail modal — functionally identical to the old `Alerts()` / `RecordsPage(mode="near")`, just co-located and now paginated (previously Alert review had `limit`/`offset` params on the backend that the frontend never used; Near miss had none). The near-miss review workflow (`Acknowledge` / `Flag for follow-up`, `POST /api/near-miss/{id}/review`) is unchanged. The free-text search that Near miss inherited from the shared `RecordsPage` component before this pass is dropped — Alert review never had one, and keeping it would have meant either searching only the current page (inconsistent with the exact-count pagination this page now has) or refetching everything, so the two review queues now share the same filter pattern: dropdown filters only, both server-side.

**3. Near-miss "Flag for follow-up" now creates a regression-test entry.** See `NEAR-MISS-FN-FP-ASSESSMENT.md` for the reasoning — short version: it closes the one real gap in the FP/FN feedback-loop idea from that pasted conversation, using the `regression_tests` mechanism that already existed for exactly this purpose, instead of introducing new status vocabulary or a parallel table.

## Verification
- Python test suite: 15 passed as of the initial redesign; two new static tests (`test_near_miss_has_own_review_workflow`, `test_history_shows_downstream_review_outcome`) were added but have not been run — left for manual verification per your instruction.
- TypeScript/TSX syntax transpilation: 0 diagnostics as of the initial redesign; subsequent edits (near-miss review workflow, history outcome tracking) have only had a manual brace-balance check, not a real build/typecheck.
- Full Vite build could not be run in the build environment because its internal npm registry returns 404 for required React type packages. No npm dependency versions were changed.
- Second pass (history pagination, merged Status review, near-miss regression-test wiring): nothing was run — no Python tests, no TSX typecheck, no build — per instruction. Manual checks only: brace/paren balance on `main.tsx`, and re-reading every touched endpoint against the existing `tests/test_local_mode.py` assertions (`"/api/near-miss/{prediction_id}/review"`, `"Near-miss record not found"`, `"decision must be acknowledged or flagged"`, no `"/api/near-miss/{prediction_id}/promote"`, `historyOutcome`/`"Alert outcome"`/`"Near-miss outcome"` in `main.tsx`, `alert_status`/`near_miss_status` in both `api/main.py` and `api/mock_main.py`) to confirm none of those literal strings moved or were removed. Please run the suite and a real build before trusting this.
