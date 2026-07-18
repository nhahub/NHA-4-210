# ============================================================
# db_connection.py
# Shared SQL Server connection manager using pyodbc
# ============================================================

import os
import pyodbc
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

SQL_SERVER   = os.getenv("SQL_SERVER",   "localhost")
SQL_DATABASE = os.getenv("SQL_DATABASE", "HeartDiseaseDB")
SQL_DRIVER   = os.getenv("SQL_DRIVER",   "ODBC Driver 17 for SQL Server")
SQL_TRUSTED  = os.getenv("SQL_TRUSTED",  "yes").lower()


def _build_connection_string() -> str:
    """Build the ODBC connection string."""
    cs = (
        f"DRIVER={{{SQL_DRIVER}}};"
        f"SERVER={SQL_SERVER};"
        f"DATABASE={SQL_DATABASE};"
    )
    if SQL_TRUSTED == "yes":
        cs += "Trusted_Connection=yes;"
    else:
        sql_user = os.getenv("SQL_USER", "")
        sql_pass = os.getenv("SQL_PASSWORD", "")
        cs += f"UID={sql_user};PWD={sql_pass};"
    return cs


@contextmanager
def get_sql_connection():
    """
    Context manager that yields a pyodbc connection.
    Usage:
        with get_sql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT ...")
    """
    conn = None
    try:
        conn = pyodbc.connect(_build_connection_string(), autocommit=False)
        yield conn
    except pyodbc.Error as e:
        if conn:
            conn.rollback()
        raise
    finally:
        if conn:
            conn.close()


def execute_sql_file(filepath: str):
    """
    Execute a SQL file containing multiple batches separated by GO.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()

    # Split on GO statements (standalone on a line)
    import re
    batches = re.split(r'^\s*GO\s*$', content, flags=re.MULTILINE | re.IGNORECASE)

    with get_sql_connection() as conn:
        cursor = conn.cursor()
        for batch in batches:
            batch = batch.strip()
            if batch and not batch.startswith("--"):
                try:
                    cursor.execute(batch)
                    conn.commit()
                except pyodbc.Error as e:
                    print(f"⚠️  Error executing batch: {e}")
                    conn.rollback()
        cursor.close()
    print(f"✅ Executed SQL file: {filepath}")
