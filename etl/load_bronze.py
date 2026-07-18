# ============================================================
# load_bronze.py
# ETL Step 2: Copy Staging → Bronze with audit metadata
# ============================================================

import uuid
import hashlib
from datetime import datetime
from db_connection import get_sql_connection


STAGING_COLUMNS = [
    "HeartDisease", "BMI", "Smoking", "AlcoholDrinking", "Stroke",
    "PhysicalHealth", "MentalHealth", "DiffWalking", "Sex", "AgeCategory",
    "Race", "Diabetic", "PhysicalActivity", "GenHealth", "SleepTime",
    "Asthma", "KidneyDisease", "SkinCancer"
]


def load_bronze(batch_id: str) -> dict:
    """
    Copy data from stg.RawPatientData → bronze.PatientData,
    adding LoadBatchID, LoadTimestamp, and RecordHash.
    """
    start = datetime.now()
    result = {"layer": "bronze", "step": "Staging → Bronze", "status": "FAILED"}

    cols_csv = ", ".join(STAGING_COLUMNS)
    cols_with_audit = f"{cols_csv}, LoadBatchID, RecordHash, RecordSource"
    
    # Build the hash expression: HASHBYTES on all columns concatenated
    hash_concat = " + '|' + ".join([f"ISNULL({c}, '')" for c in STAGING_COLUMNS])
    hash_expr = f"CONVERT(VARCHAR(64), HASHBYTES('SHA2_256', {hash_concat}), 2)"

    insert_sql = f"""
        INSERT INTO bronze.PatientData ({cols_with_audit})
        SELECT
            {cols_csv},
            '{batch_id}' AS LoadBatchID,
            {hash_expr} AS RecordHash,
            'heart_2020_cleaned.csv' AS RecordSource
        FROM stg.RawPatientData
    """

    with get_sql_connection() as conn:
        cursor = conn.cursor()

        # Count staging rows
        cursor.execute("SELECT COUNT(*) FROM stg.RawPatientData")
        staging_count = cursor.fetchone()[0]

        if staging_count == 0:
            result["error"] = "Staging table is empty — run load_staging first"
            return result

        print(f"📦 Copying {staging_count:,} rows from Staging → Bronze...")

        # Insert with hash
        cursor.execute(insert_sql)
        inserted = cursor.rowcount
        conn.commit()
        cursor.close()

    end = datetime.now()
    duration = (end - start).total_seconds()

    print(f"✅ Bronze load complete: {inserted:,} rows in {duration:.1f}s")

    _log_step(batch_id, "bronze", "Staging → Bronze", staging_count, inserted, 0, start, end, "SUCCESS")

    result.update({
        "status": "SUCCESS",
        "rows_processed": staging_count,
        "rows_inserted": inserted,
        "duration_seconds": duration
    })
    return result


def _log_step(batch_id, layer, step, processed, inserted, rejected, start, end, status, error=None):
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
    load_bronze(bid)
