"""
Postgres connection + incremental ("watermark") row fetch for
predict_realtime.py.

Reads connection info + table name from environment variables (.env,
gitignored — see .env.example), with optional CLI flags as overrides.
Never hardcode credentials here.

Watermark strategy: the caller (predict_realtime.read_sensor) tracks the
timestamp of the last row it has already yielded and passes it back in as
`since`. Each call returns only rows strictly newer than that, ordered
ascending, so the rolling-window / Kalman pipeline downstream keeps
seeing ticks in order with no gaps or repeats. On the very first call
(since=None) everything currently in the table is returned, mirroring
the old CSV replay behavior.
"""

import argparse
import os

import psycopg2
import psycopg2.extras
import psycopg2.sql as sql

import config

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    # python-dotenv not installed — fine if the env vars are already set
    # some other way (shell export, systemd EnvironmentFile, etc).
    pass

DEFAULT_MACHINE_ID = os.environ.get("DEFAULT_MACHINE_ID", "MACHINE-001").strip() or "MACHINE-001"


def parse_args():
    """
    CLI overrides for the env vars below. Anything not passed on the
    command line falls back to the corresponding PG_* env var.
    Usage: python predict_realtime.py --host ... --user ... --password ...
    """
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--host")
    parser.add_argument("--port")
    parser.add_argument("--db", dest="database")
    parser.add_argument("--user")
    parser.add_argument("--password")
    parser.add_argument("--table")
    args, _unknown = parser.parse_known_args()
    return args


def _env_only_args():
    """Return empty DB CLI overrides for library/server callers.

    Uvicorn and other host processes have their own --port/--host flags.
    Parsing sys.argv here would accidentally treat those as PostgreSQL
    settings (for example uvicorn --port 8000 becoming PG_PORT=8000).
    Only predict_realtime.py explicitly calls parse_args() and passes the
    result into get_connection/get_table_name.
    """
    return argparse.Namespace(host=None, port=None, database=None, user=None, password=None, table=None)


def get_connection(args=None):
    args = args if args is not None else _env_only_args()

    host = args.host or os.environ.get("PG_HOST")
    port = args.port or os.environ.get("PG_PORT", "5432")
    database = args.database or os.environ.get("PG_DATABASE")
    user = args.user or os.environ.get("PG_USER")
    password = args.password or os.environ.get("PG_PASSWORD")

    missing = [name for name, val in [
        ("host", host), ("database", database), ("user", user), ("password", password)
    ] if not val]
    if missing:
        raise ValueError(
            f"Missing Postgres connection info: {', '.join(missing)}. "
            f"Set PG_HOST/PG_DATABASE/PG_USER/PG_PASSWORD in .env, or pass "
            f"--host/--db/--user/--password."
        )

    return psycopg2.connect(host=host, port=port, dbname=database, user=user, password=password)


def get_table_name(args=None):
    args = args if args is not None else _env_only_args()
    table = args.table or os.environ.get("PG_TABLE")
    if not table:
        raise ValueError("Missing table name. Set PG_TABLE in .env, or pass --table.")
    return table


def get_db_columns():
    """
    Column names to select from the Postgres table. Defaults to
    config.py's CSV column names for config.RAW_SENSOR_COLS (confirmed to
    match); override any individual one via an env var named
    PG_COL_<COLUMN_NAME_UPPERCASED> in .env if the DB table ever names a
    column differently — e.g. config.COL_A_RMS = "a_rms_mps2" is
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
    query = sql.SQL(
        "SELECT DISTINCT {machine} FROM {table} WHERE {machine} IS NOT NULL "
        "ORDER BY {machine} LIMIT %s"
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
    huge backlog — e.g. after downtime — flooding memory in one go; the
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


def fetch_rows_between(conn, table: str, start, end, limit: int | None = None, machine_id: str | None = None):
    """
    Returns rows with start <= timestamp < end, ordered ascending, as a
    list of dicts keyed by the Postgres column names from
    get_db_columns() — same row shape as fetch_new_rows().

    Used by train_isolation_forest.py to pull an explicit, pinned
    commissioning/reference window directly from the same table +
    connection worker.py polls in production, instead of a separate
    offline CSV (see config.REFERENCE_SOURCE). Unlike fetch_new_rows()'s
    open-ended "everything newer than since", this always takes a closed
    [start, end) range, so re-running training reads exactly the same
    rows every time regardless of how much the table has grown since —
    the range itself is what makes a live-sourced training run
    reproducible, the same way a static CSV file's fixed content used to.

    The default is intentionally uncapped. A previous 200,000-row default
    silently truncated dense commissioning windows; several machines could
    then train on different incomplete time spans. Callers may still provide
    an explicit positive limit for diagnostic use.
    """
    dbcols = get_db_columns()
    cols = [dbcols["timestamp"]] + ([dbcols["machine_id"]] if dbcols["machine_id"] else []) + dbcols["sensor_cols"]
    col_ident = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
    table_ident = sql.Identifier(table)
    ts_ident = sql.Identifier(dbcols["timestamp"])

    machine_column = dbcols["machine_id"]
    if machine_id and machine_column:
        query = sql.SQL(
            "SELECT {cols} FROM {table} WHERE {ts} >= %s AND {ts} < %s "
            "AND {machine} = %s ORDER BY {ts} ASC"
        ).format(
            cols=col_ident, table=table_ident, ts=ts_ident,
            machine=sql.Identifier(machine_column),
        )
        args = [start, end, machine_id]
    elif machine_id and machine_id != DEFAULT_MACHINE_ID:
        return []
    else:
        query = sql.SQL(
            "SELECT {cols} FROM {table} WHERE {ts} >= %s AND {ts} < %s ORDER BY {ts} ASC"
        ).format(cols=col_ident, table=table_ident, ts=ts_ident)
        args = [start, end]
    if limit is not None:
        if int(limit) < 1:
            raise ValueError("limit must be positive when provided")
        query += sql.SQL(" LIMIT %s")
        args.append(int(limit))
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, tuple(args))
        return cur.fetchall()


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
            "WITH ranked AS ("
            " SELECT {columns}, row_number() OVER (PARTITION BY {machine} ORDER BY {timestamp} DESC) AS source_rank"
            " FROM {table}"
            ") SELECT {columns} FROM ranked WHERE source_rank <= %s "
            "ORDER BY {timestamp}, {machine}"
        ).format(
            columns=column_identifiers,
            machine=machine_identifier,
            timestamp=timestamp_identifier,
            table=table_identifier,
        )
    else:
        query = sql.SQL(
            "SELECT {columns} FROM (SELECT {columns} FROM {table} "
            "ORDER BY {timestamp} DESC LIMIT %s) recent ORDER BY {timestamp}"
        ).format(columns=column_identifiers, table=table_identifier, timestamp=timestamp_identifier)

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(query, (rows_per_machine,))
        return cur.fetchall()

# End of database helpers.
