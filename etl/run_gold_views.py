# ============================================================
# run_gold_views.py
# ETL Step 4: Deploy Gold Layer Star Schema Views
# ============================================================

import os
import uuid
from datetime import datetime
from db_connection import execute_sql_file, get_sql_connection


SQL_DIR = os.path.join(os.path.dirname(__file__), "..", "sql")


def run_gold_views(batch_id: str) -> dict:
    """
    Execute the Gold Layer SQL file to create/update Star Schema views.
    """
    start = datetime.now()
    result = {"layer": "gold", "step": "Deploy Gold Views", "status": "FAILED"}

    gold_sql = os.path.join(SQL_DIR, "05_GoldLayer.sql")

    if not os.path.exists(gold_sql):
        result["error"] = f"Gold SQL file not found: {gold_sql}"
        return result

    print("🏆 Deploying Gold Layer Star Schema Views...")

    try:
        execute_sql_file(gold_sql)

        # Verify views were created
        with get_sql_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT TABLE_NAME
                FROM INFORMATION_SCHEMA.VIEWS
                WHERE TABLE_SCHEMA = 'gold'
                ORDER BY TABLE_NAME
            """)
            views = [row[0] for row in cursor.fetchall()]
            cursor.close()

        print(f"   ✅ Gold views deployed: {', '.join(views)}")
        view_count = len(views)

    except Exception as e:
        result["error"] = str(e)
        _log_step(batch_id, "gold", "Deploy Gold Views", 0, 0, 0, start, datetime.now(), "FAILED", str(e))
        return result

    end = datetime.now()
    duration = (end - start).total_seconds()

    _log_step(batch_id, "gold", "Deploy Gold Views", view_count, view_count, 0, start, end, "SUCCESS")

    result.update({
        "status": "SUCCESS",
        "views_created": views,
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
    run_gold_views(bid)
