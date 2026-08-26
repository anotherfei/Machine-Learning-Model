"""PostgreSQL connection, source mapping, and ordered sensor-row access."""

from datetime import timedelta
import os

import psycopg2
import psycopg2.extras
import psycopg2.sql as sql

import config

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # python-dotenv not installed - fine if the env vars are already set
    # some other way (shell export, systemd EnvironmentFile, etc).
    pass

DEFAULT_MACHINE_ID = os.environ.get("DEFAULT_MACHINE_ID", "MACHINE-001").strip() or "MACHINE-001"


def get_connection():
    host = os.environ.get("PG_HOST")
    port = os.environ.get("PG_PORT", "5432")
    database = os.environ.get("PG_DATABASE")
    user = os.environ.get("PG_USER")
    password = os.environ.get("PG_PASSWORD")

    missing = [name for name, val in [
        ("host", host), ("database", database), ("user", user), ("password", password)
    ] if not val]
    if missing:
        raise ValueError(
            f"Missing Postgres connection info: {', '.join(missing)}. "
            "Set PG_HOST/PG_DATABASE/PG_USER/PG_PASSWORD in .env."
        )

    connection = psycopg2.connect(
        host=host,
        port=port,
        dbname=database,
        user=user,
        password=password,
        connect_timeout=10,
    )
    # Keep the application's compact control plane isolated from the existing
    # production sensor schema. PostgreSQL accepts a not-yet-created schema in
    # search_path, so the first connection can still run db_schema.migrate().
    connection.autocommit = True
    with connection.cursor() as cursor:
        cursor.execute('SET search_path TO "ML", public')
    connection.autocommit = False
    return connection


def get_table_name():
    table = os.environ.get("PG_TABLE")
    if not table:
        raise ValueError("Missing table name. Set PG_TABLE in .env.")
    return table


def get_db_columns():
    """
    Column names to select from the Postgres table. Defaults to
    config.py's CSV column names for config.RAW_SENSOR_COLS (confirmed to
    match); override any individual one via an env var named
    PG_COL_<COLUMN_NAME_UPPERCASED> in .env if the DB table ever names a
    column differently - e.g. config.COL_A_RMS = "a_rms_mps2" is
    overridden by PG_COL_A_RMS_MPS2. Generic over however many/whatever
    raw sensor columns config.py currently defines, so a future sensor
    swap only means editing config.RAW_SENSOR_COLS, not this function.
    """
    timestamp = os.environ.get(f"PG_COL_{config.COL_TIMESTAMP.upper()}", config.COL_TIMESTAMP)
    # Optional for backwards compatibility with single-machine source tables.
    # Set PG_COL_MACHINE_ID to the source column name to enable multi-machine
    # polling. When it is unset, every row belongs to DEFAULT_MACHINE_ID.
    machine_id = os.environ.get("PG_COL_MACHINE_ID", "").strip() or None
    sensor_cols = [
        os.environ.get(f"PG_COL_{col.upper()}", col) for col in config.RAW_SENSOR_COLS
    ]
    # Maps each config.py raw column name -> its actual Postgres column
    # name, so callers can look up "give me whatever DB column holds
    # config.COL_A_RMS" without knowing if it was overridden.
    by_config_name = dict(zip(config.RAW_SENSOR_COLS, sensor_cols))
    return {
        "timestamp": timestamp,
        "machine_id": machine_id,
        "sensor_cols": sensor_cols,
        "by_config_name": by_config_name,
    }


def row_machine_id(row, dbcols=None) -> str:
    """Return a normalized machine ID for a source row."""
    dbcols = dbcols or get_db_columns()
    column = dbcols["machine_id"]
    value = row.get(column) if column else DEFAULT_MACHINE_ID
    value = str(value).strip() if value is not None else ""
    return value or DEFAULT_MACHINE_ID


def canonical_sensor_reading(row, dbcols=None) -> dict:
    """Map one PostgreSQL source row into config.py's canonical units.

    PostgreSQL remains an immutable source of truth.  Unit conversion belongs
    here so training, realtime inference, backfill, retraining, and the API
    cannot accidentally interpret the same source column differently.
    Missing values are preserved for preprocessing/state validation to handle.
    """
    dbcols = dbcols or get_db_columns()
    reading = {}
    for config_name, db_name in dbcols["by_config_name"].items():
        value = row[db_name]
        reading[config_name] = (
            None if value is None
            else float(value) * float(config.POSTGRES_SENSOR_SCALES.get(config_name, 1.0))
        )
    return reading


def canonical_sensor_frame(rows, dbcols=None):
    """Return PostgreSQL rows as a canonical timestamp + sensor DataFrame."""
    import pandas as pd

    dbcols = dbcols or get_db_columns()
    records = [
        {
            config.COL_TIMESTAMP: row[dbcols["timestamp"]],
            **canonical_sensor_reading(row, dbcols),
        }
        for row in rows
    ]
    return pd.DataFrame(records, columns=[config.COL_TIMESTAMP, *config.RAW_SENSOR_COLS])


def fetch_machine_ids(conn, table: str, limit: int = 1000) -> list[str]:
    """Discover asset IDs from the sensor source table.

    VVB001 identifies the shared sensor model/specification; this returns
    values from the separate asset identity column configured through
    PG_COL_MACHINE_ID. Legacy single-machine tables return the configured
    DEFAULT_MACHINE_ID.
    """
    machine_column = get_db_columns()["machine_id"]
    if not machine_column:
        return [DEFAULT_MACHINE_ID]
    # PostgreSQL normally implements SELECT DISTINCT by walking every entry in
    # the source index. On a tens-of-millions-row sensor table that can exceed
    # the website timeout even when there are only four machines. This loose
    # index scan repeatedly seeks to the first value greater than the previous
    # one, so work scales with the number of machines rather than source rows.
    query = sql.SQL(
        "WITH RECURSIVE discovered(machine_id) AS ("
        " (SELECT {machine} FROM {table} WHERE {machine} IS NOT NULL "
        "  ORDER BY {machine} LIMIT 1)"
        " UNION ALL"
        " SELECT (SELECT {machine} FROM {table} "
        "         WHERE {machine} > discovered.machine_id "
        "           AND {machine} IS NOT NULL "
        "         ORDER BY {machine} LIMIT 1)"
        " FROM discovered WHERE discovered.machine_id IS NOT NULL"
        ") SELECT machine_id FROM discovered "
        "WHERE machine_id IS NOT NULL LIMIT %s"
    ).format(
        machine=sql.Identifier(machine_column),
        table=sql.Identifier(table),
    )
    with conn.cursor() as cur:
        cur.execute(query, (max(1, min(limit, 10000)),))
        return [str(row[0]).strip() for row in cur.fetchall() if str(row[0]).strip()]


def fetch_latest_row(conn, table: str, machine_id: str):
    """Return the newest raw source row for one machine, or None.

    This is intentionally independent of the inference worker. The website
    uses it to prove that it is connected to the configured production source
    and to display current sensor channels even while a model is warming up or
    the worker is unavailable.
    """
    dbcols = get_db_columns()
    columns = [dbcols["timestamp"]] + ([dbcols["machine_id"]] if dbcols["machine_id"] else []) + dbcols["sensor_cols"]
    column_identifiers = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    table_identifier = sql.Identifier(table)
    timestamp_identifier = sql.Identifier(dbcols["timestamp"])

    if dbcols["machine_id"]:
        query = sql.SQL(
            "SELECT {columns} FROM {table} WHERE {machine} = %s "
            "ORDER BY {timestamp} DESC LIMIT 1"
        ).format(
            columns=column_identifiers,
            table=table_identifier,
            machine=sql.Identifier(dbcols["machine_id"]),
            timestamp=timestamp_identifier,
        )
        args = (machine_id,)
    elif machine_id != DEFAULT_MACHINE_ID:
        return None
    else:
        query = sql.SQL(
            "SELECT {columns} FROM {table} ORDER BY {timestamp} DESC LIMIT 1"
        ).format(columns=column_identifiers, table=table_identifier, timestamp=timestamp_identifier)
        args = ()

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, args)
        return cur.fetchone()


def fetch_new_rows(conn, table: str, since=None, limit: int = 5000):
    """
    Returns rows after `since` (or all rows when it is None), ordered by
    timestamp and machine ID. Multi-machine sources use a composite
    `(timestamp, machine_id)` cursor; legacy sources use a timestamp cursor.

    limit caps how many rows come back in one poll (protects against a
    huge backlog - e.g. after downtime - flooding memory in one go; the
    watermark just picks up where it left off on the next 60s poll).
    """
    dbcols = get_db_columns()
    cols = [dbcols["timestamp"]] + ([dbcols["machine_id"]] if dbcols["machine_id"] else []) + dbcols["sensor_cols"]
    col_ident = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
    table_ident = sql.Identifier(table)
    ts_ident = sql.Identifier(dbcols["timestamp"])

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if since is None:
            order = sql.SQL("{ts}, {machine}").format(
                ts=ts_ident,
                machine=sql.Identifier(dbcols["machine_id"]),
            ) if dbcols["machine_id"] else ts_ident
            query = sql.SQL("SELECT {cols} FROM {table} ORDER BY {order} ASC LIMIT %s").format(
                cols=col_ident, table=table_ident, order=order
            )
            cur.execute(query, (limit,))
        elif dbcols["machine_id"]:
            # Composite cursor prevents rows for a second machine at the same
            # timestamp from being skipped when a poll is split by `limit`.
            since_ts, since_machine = since
            machine_ident = sql.Identifier(dbcols["machine_id"])
            query = sql.SQL(
                "SELECT {cols} FROM {table} WHERE ({ts}, {machine}) > (%s, %s) "
                "ORDER BY {ts}, {machine} ASC LIMIT %s"
            ).format(cols=col_ident, table=table_ident, ts=ts_ident, machine=machine_ident)
            cur.execute(query, (since_ts, since_machine, limit))
        else:
            query = sql.SQL(
                "SELECT {cols} FROM {table} WHERE {ts} > %s ORDER BY {ts} ASC LIMIT %s"
            ).format(cols=col_ident, table=table_ident, ts=ts_ident)
            cur.execute(query, (since, limit))
        return cur.fetchall()


def count_rows_between(conn, table: str, start, end, machine_ids=None) -> int:
    """Count a bounded source range using the production source index."""
    dbcols = get_db_columns()
    table_ident = sql.Identifier(table)
    ts_ident = sql.Identifier(dbcols["timestamp"])
    machine_column = dbcols["machine_id"]
    if machine_ids and machine_column:
        query = sql.SQL(
            "SELECT count(*) FROM {table} "
            "WHERE {machine}=ANY(%s) AND {ts}>=%s AND {ts}<%s"
        ).format(
            table=table_ident,
            machine=sql.Identifier(machine_column),
            ts=ts_ident,
        )
        args = (list(machine_ids), start, end)
    elif machine_ids and not machine_column and DEFAULT_MACHINE_ID not in machine_ids:
        return 0
    else:
        query = sql.SQL(
            "SELECT count(*) FROM {table} WHERE {ts}>=%s AND {ts}<%s"
        ).format(table=table_ident, ts=ts_ident)
        args = (start, end)
    with conn.cursor() as cur:
        cur.execute(query, args)
        return int(cur.fetchone()[0])


def iter_row_chunks_between(
    table: str,
    start,
    end,
    *,
    machine_id: str | None = None,
    chunk_rows: int | None = None,
    slice_hours: int | None = None,
    statement_timeout_ms: int | None = None,
):
    """Yield an indexed source range without materializing it in memory.

    A dedicated read-only connection and named server-side cursor keep both
    libpq and Python memory bounded.  Short half-open time slices release old
    MVCC snapshots regularly, provide visible progress to the caller, and make
    cancellation reliable on Windows even when a very large range is selected.
    """
    chunk_rows = int(chunk_rows or config.TRAINING_DB_CHUNK_ROWS)
    slice_hours = int(slice_hours or config.TRAINING_DB_SLICE_HOURS)
    statement_timeout_ms = int(
        statement_timeout_ms or config.TRAINING_DB_STATEMENT_TIMEOUT_MS
    )
    if chunk_rows < 1:
        raise ValueError("chunk_rows must be positive")
    if slice_hours < 1:
        raise ValueError("slice_hours must be positive")
    if statement_timeout_ms < 1:
        raise ValueError("statement_timeout_ms must be positive")
    if end <= start:
        raise ValueError("end must be after start")

    dbcols = get_db_columns()
    cols = [dbcols["timestamp"]] + (
        [dbcols["machine_id"]] if dbcols["machine_id"] else []
    ) + dbcols["sensor_cols"]
    col_ident = sql.SQL(", ").join(sql.Identifier(column) for column in cols)
    table_ident = sql.Identifier(table)
    ts_ident = sql.Identifier(dbcols["timestamp"])
    machine_column = dbcols["machine_id"]
    if machine_id and not machine_column and machine_id != DEFAULT_MACHINE_ID:
        return

    conn = get_connection()
    try:
        conn.set_session(readonly=True, autocommit=False)
        slice_start = start
        slice_number = 0
        while slice_start < end:
            slice_end = min(end, slice_start + timedelta(hours=slice_hours))
            slice_number += 1
            try:
                with conn.cursor() as settings_cursor:
                    settings_cursor.execute(
                        "SELECT set_config('statement_timeout', %s, true)",
                        (f"{statement_timeout_ms}ms",),
                    )

                cursor_name = f"training_source_{os.getpid()}_{slice_number}"
                with conn.cursor(
                    name=cursor_name,
                    cursor_factory=psycopg2.extras.RealDictCursor,
                ) as cur:
                    cur.itersize = chunk_rows
                    if machine_id and machine_column:
                        query = sql.SQL(
                            "SELECT {cols} FROM {table} "
                            "WHERE {machine} = %s AND {ts} >= %s AND {ts} < %s "
                            "ORDER BY {ts} ASC"
                        ).format(
                            cols=col_ident,
                            table=table_ident,
                            machine=sql.Identifier(machine_column),
                            ts=ts_ident,
                        )
                        args = (machine_id, slice_start, slice_end)
                    else:
                        query = sql.SQL(
                            "SELECT {cols} FROM {table} "
                            "WHERE {ts} >= %s AND {ts} < %s ORDER BY {ts} ASC"
                        ).format(cols=col_ident, table=table_ident, ts=ts_ident)
                        args = (slice_start, slice_end)
                    cur.execute(query, args)
                    while True:
                        rows = cur.fetchmany(chunk_rows)
                        if not rows:
                            break
                        yield rows
                conn.rollback()
            except KeyboardInterrupt:
                try:
                    conn.cancel()
                    conn.rollback()
                except Exception:
                    pass
                raise
            except psycopg2.errors.QueryCanceled as exc:
                conn.rollback()
                raise TimeoutError(
                    f"PostgreSQL training read timed out for machine "
                    f"{machine_id or DEFAULT_MACHINE_ID!r} in slice "
                    f"[{slice_start}, {slice_end}). Verify the composite "
                    "(machine_id, timestamp) source index or increase "
                    "TRAINING_DB_STATEMENT_TIMEOUT_MS."
                ) from exc
            slice_start = slice_end
    finally:
        conn.close()


def fetch_rows_before(conn, table: str, before, limit: int, machine_id: str | None = None):
    """Return the exact preceding source rows in chronological order.

    Rolling features need a row-count warm-up, not an elapsed-time estimate:
    real industrial feeds can be delayed or sparse even when a nominal sample
    rate is configured.
    """
    limit = int(limit)
    if limit < 1:
        raise ValueError("limit must be positive")
    dbcols = get_db_columns()
    cols = [dbcols["timestamp"]] + ([dbcols["machine_id"]] if dbcols["machine_id"] else []) + dbcols["sensor_cols"]
    col_ident = sql.SQL(", ").join(sql.Identifier(column) for column in cols)
    table_ident = sql.Identifier(table)
    ts_ident = sql.Identifier(dbcols["timestamp"])
    if machine_id and dbcols["machine_id"]:
        query = sql.SQL(
            "SELECT {cols} FROM {table} WHERE {ts}<%s AND {machine}=%s "
            "ORDER BY {ts} DESC LIMIT %s"
        ).format(
            cols=col_ident, table=table_ident, ts=ts_ident,
            machine=sql.Identifier(dbcols["machine_id"]),
        )
        args = (before, machine_id, limit)
    elif machine_id and machine_id != DEFAULT_MACHINE_ID:
        return []
    else:
        query = sql.SQL(
            "SELECT {cols} FROM {table} WHERE {ts}<%s ORDER BY {ts} DESC LIMIT %s"
        ).format(cols=col_ident, table=table_ident, ts=ts_ident)
        args = (before, limit)
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, args)
        return list(reversed(cur.fetchall()))

def fetch_recent_rows(conn, table: str, rows_per_machine: int = 30):
    """Return a recent warm-up tail for every machine in chronological order.

    A production worker restart should synchronize to the current source, not
    replay the table from its oldest row. Each machine gets enough independent
    history to rebuild rolling/Kalman state before incremental polling resumes.
    Multi-machine sources are queried one machine at a time so PostgreSQL can
    stop after ``rows_per_machine`` entries in the existing
    ``(machine_id, timestamp)`` index. A partitioned ``row_number()`` query
    would rank the entire source table before filtering and can stall startup
    for minutes on production tables containing tens of millions of rows.
    """
    rows_per_machine = max(1, min(int(rows_per_machine), 10000))
    dbcols = get_db_columns()
    columns = [dbcols["timestamp"]] + ([dbcols["machine_id"]] if dbcols["machine_id"] else []) + dbcols["sensor_cols"]
    column_identifiers = sql.SQL(", ").join(sql.Identifier(column) for column in columns)
    timestamp_identifier = sql.Identifier(dbcols["timestamp"])
    table_identifier = sql.Identifier(table)

    if dbcols["machine_id"]:
        machine_identifier = sql.Identifier(dbcols["machine_id"])
        query = sql.SQL(
            "SELECT {columns} FROM {table} WHERE {machine} = %s "
            "ORDER BY {timestamp} DESC LIMIT %s"
        ).format(
            columns=column_identifiers,
            machine=machine_identifier,
            timestamp=timestamp_identifier,
            table=table_identifier,
        )
        machine_ids = fetch_machine_ids(conn, table, limit=10000)
        rows = []
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            for machine_id in machine_ids:
                cur.execute(query, (machine_id, rows_per_machine))
                rows.extend(cur.fetchall())
        rows.sort(key=lambda row: (row[dbcols["timestamp"]], str(row[dbcols["machine_id"]])))
        return rows
    else:
        query = sql.SQL(
            "SELECT {columns} FROM (SELECT {columns} FROM {table} "
            "ORDER BY {timestamp} DESC LIMIT %s) recent ORDER BY {timestamp}"
        ).format(columns=column_identifiers, table=table_identifier, timestamp=timestamp_identifier)

        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(query, (rows_per_machine,))
            return cur.fetchall()

# End of database helpers.
