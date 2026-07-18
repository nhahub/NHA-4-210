# ============================================================
# load_staging.py
# ETL Step 1: Load raw CSV data into stg.RawPatientData
# ============================================================

import os
import uuid
import pandas as pd
from datetime import datetime
from db_connection import get_sql_connection

CSV_PATH = os.getenv(
    "CSV_SOURCE_PATH",
    os.path.join(os.path.dirname(__file__), "..", "data", "heart_2020_cleaned.csv")
)

# Column mapping: CSV column name → Staging column name
STAGING_COLUMNS = [
    "HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke",
    "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex", "AgeCategory",
    "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime",
    "Asthma", "KidneyDisease", "SkinCancer"
]


def load_staging(batch_id: str) -> dict:
    """
    Read the raw CSV and bulk-insert into stg.RawPatientData.
    Returns a summary dict with row counts and timing.
    """
    start = datetime.now()
    result = {"layer": "staging", "step": "CSV → Staging", "status": "FAILED"}

    # ── 1. Read CSV ─────────────────────────────────────────
    csv_path = os.path.abspath(CSV_PATH)
    if not os.path.exists(csv_path):
        result["error"] = f"CSV file not found: {csv_path}"
        return result

    print(f"📄 Reading CSV: {csv_path}")
    df = pd.read_csv(csv_path, dtype=str)  # Read everything as string
    total_rows = len(df)
    print(f"   → {total_rows:,} rows read from CSV")

    # Ensure columns match
    missing = [c for c in STAGING_COLUMNS if c not in df.columns]
    if missing:
        result["error"] = f"Missing columns in CSV: {missing}"
        return result

    df = df[STAGING_COLUMNS]

    # ── 2. Truncate staging table ───────────────────────────
    with get_sql_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE stg.RawPatientData")
        conn.commit()
        print("🗑️  Staging table truncated")

        # ── 3. Bulk insert ──────────────────────────────────
        placeholders = ", ".join(["?"] * len(STAGING_COLUMNS))
        insert_sql = f"INSERT INTO stg.RawPatientData ({', '.join(STAGING_COLUMNS)}) VALUES ({placeholders})"

        batch_size = 5000
        inserted = 0

        for i in range(0, total_rows, batch_size):
            chunk = df.iloc[i:i + batch_size]
            rows = [tuple(row) for row in chunk.values]
            cursor.executemany(insert_sql, rows)
            conn.commit()
            inserted += len(rows)
            pct = round(inserted / total_rows * 100, 1)
            print(f"   ✅ Inserted {inserted:,} / {total_rows:,} ({pct}%)", end="\r")

        cursor.close()

    end = datetime.now()
    duration = (end - start).total_seconds()

    print(f"\n✅ Staging load complete: {inserted:,} rows in {duration:.1f}s")

    # ── 4. Log the result ───────────────────────────────────
    _log_step(batch_id, "staging", "CSV → Staging", total_rows, inserted, 0, start, end, "SUCCESS")

    result.update({
        "status": "SUCCESS",
        "rows_processed": total_rows,
        "rows_inserted": inserted,
        "duration_seconds": duration
    })
    return result


def _log_step(batch_id, layer, step, processed, inserted, rejected, start, end, status, error=None):
    """Log ETL step to ops.PipelineLog."""
    try:
        with get_sql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "EXEC ops.sp_LogStep @BatchID=?, @LayerName=?, @StepName=?, "
                "@RowsProcessed=?, @RowsInserted=?, @RowsRejected=?, "
                "@Status=?, @ErrorMessage=?, @StartTime=?, @EndTime=?",
                batch_id, layer, step, processed, inserted, rejected,
                status, error, start, end
            )
            conn.commit()
            cursor.close()
    except Exception as e:
        print(f"⚠️  Logging failed: {e}")


if __name__ == "__main__":
    bid = str(uuid.uuid4())
    print(f"🔑 Batch ID: {bid}")
    load_staging(bid)
