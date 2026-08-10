# Frontend redesign notes

The React frontend was rebuilt around the existing backend contracts. No new pages were added.

## Main changes
- Replaced the prototype card grid with a unified machine-status board and compact live sensor strip.
- Added collapsible navigation, top status bar, responsive layout, and redesigned login experience.
- Added interactive drill-down modals/popovers instead of adding more permanent containers.
- Dashboard: status evidence, model/runtime context, mock-mode explanation, sensor channel details, anomaly explanation, live sparklines.
- Alerts: filters, structured table, row inspection, raw sensor disclosure, ±3-hour context, review actions inside the detail modal.
- Models: retraining summary, structured model list, validation report detail, promotion action in the model modal.
- History / Near Miss: search and state filters, structured records table, click-to-inspect sensor/failure-probability/full-backend details.
- Thresholds: readable policy labels with backend keys preserved, hot-reload action, collapsible policy explanation.
- Environment: primary DB fields first, advanced values collapsed, credential handling explanation.
- Added responsive behavior for tablet/mobile widths.

The frontend remains presentation-only: ML, anomaly scoring, maintenance decisions, threshold validation, retraining, and database operations stay in the Python backend/worker.

## Verification
- Python test suite: 15 passed.
- TypeScript/TSX syntax transpilation: 0 diagnostics.
- Full Vite build could not be run in the build environment because its internal npm registry returns 404 for required React type packages. No npm dependency versions were changed.
