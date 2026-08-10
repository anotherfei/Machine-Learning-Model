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
    load_dotenv()
except ImportError:
    # python-dotenv not installed — fine if the env vars are already set
    # some other way (shell export, systemd EnvironmentFile, etc).
    pass


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


def get_connection(args=None):
    args = args or parse_args()

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
    args = args or parse_args()
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
    sensor_cols = [
        os.environ.get(f"PG_COL_{col.upper()}", col) for col in config.RAW_SENSOR_COLS
    ]
    # Maps each config.py raw column name -> its actual Postgres column
    # name, so callers can look up "give me whatever DB column holds
    # config.COL_A_RMS" without knowing if it was overridden.
    by_config_name = dict(zip(config.RAW_SENSOR_COLS, sensor_cols))
    return {
        "timestamp": timestamp,
        "sensor_cols": sensor_cols,
        "by_config_name": by_config_name,
    }


def fetch_new_rows(conn, table: str, since=None, limit: int = 5000):
    """
    Returns rows with timestamp > since (or all rows, if since is None),
    ordered ascending by timestamp, as a list of dicts keyed by the
    Postgres column names from get_db_columns().

    limit caps how many rows come back in one poll (protects against a
    huge backlog — e.g. after downtime — flooding memory in one go; the
    watermark just picks up where it left off on the next 60s poll).
    """
    dbcols = get_db_columns()
    cols = [dbcols["timestamp"]] + dbcols["sensor_cols"]
    col_ident = sql.SQL(", ").join(sql.Identifier(c) for c in cols)
    table_ident = sql.Identifier(table)
    ts_ident = sql.Identifier(dbcols["timestamp"])

    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        if since is None:
            query = sql.SQL("SELECT {cols} FROM {table} ORDER BY {ts} ASC LIMIT %s").format(
                cols=col_ident, table=table_ident, ts=ts_ident
            )
            cur.execute(query, (limit,))
        else:
            query = sql.SQL(
                "SELECT {cols} FROM {table} WHERE {ts} > %s ORDER BY {ts} ASC LIMIT %s"
            ).format(cols=col_ident, table=table_ident, ts=ts_ident)
            cur.execute(query, (since, limit))
        return cur.fetchall()
