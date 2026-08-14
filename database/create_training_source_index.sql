-- Run this once against the production PostgreSQL database before loading
-- long commissioning windows.  The trainer filters by mach_id, restricts a
-- datetime range, and returns rows in datetime order; this composite index
-- matches that access pattern exactly.
--
-- IMPORTANT:
--   * Run this file outside an explicit BEGIN/COMMIT transaction.
--   * CONCURRENTLY avoids blocking normal inserts/reads, but building the
--     index on a very large table can take time and use disk/CPU resources.
--   * If your PG_TABLE / PG_COL_* mappings differ, replace the identifiers.

CREATE INDEX CONCURRENTLY IF NOT EXISTS
    p1_sel5_vibration_mach_id_datetime_idx
ON p1_sel5_vibration (mach_id, datetime);

-- Refresh planner statistics after the index has been built.
ANALYZE p1_sel5_vibration (mach_id, datetime);

-- Optional read-only verification.  The plan should mention
-- p1_sel5_vibration_mach_id_datetime_idx (or an equivalent composite index),
-- not a sequential scan of the whole table.
EXPLAIN
SELECT datetime, arms, vrms, apeak, crest, temp
FROM p1_sel5_vibration
WHERE mach_id = 'B16-MTV-06'
  AND datetime >= TIMESTAMPTZ '2025-06-30T00:00:00Z'
  AND datetime <  TIMESTAMPTZ '2025-11-30T00:00:00Z'
ORDER BY datetime ASC;
