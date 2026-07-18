# ============================================================
# load_silver.py
# ETL Step 3: Clean & transform Bronze → Silver
# Type casting, business rules, derived columns
# ============================================================

import uuid
from datetime import datetime
from db_connection import get_sql_connection


def load_silver(batch_id: str) -> dict:
    """
    Transform bronze.PatientData → silver.CleanedPatientData.
    Applies type casting, data cleaning, and feature engineering.
    """
    start = datetime.now()
    result = {"layer": "silver", "step": "Bronze → Silver", "status": "FAILED"}

    # ── Build the transformation SQL ────────────────────────
    # This single INSERT...SELECT does all cleaning in one pass
    transform_sql = f"""
        INSERT INTO silver.CleanedPatientData (
            PatientID,
            HeartDisease, HeartDisease_Binary,
            Sex, AgeCategory, Age_Numeric, Age_Group, Race,
            BMI, BMI_Category, GenHealth, Health_Score,
            PhysicalHealth, MentalHealth,
            SleepTime, Sleep_Quality, DiffWalking,
            Smoking, AlcoholDrinking, PhysicalActivity,
            Stroke, Diabetic, Asthma, KidneyDisease, SkinCancer,
            Comorbidity_Count, Comorbidity_Level,
            LoadBatchID
        )
        SELECT
            -- PatientID: generate a unique key (row number based)
            'P' + RIGHT('000000' + CAST(ROW_NUMBER() OVER (ORDER BY BronzeID) AS VARCHAR), 6) AS PatientID,

            -- Core diagnosis
            HeartDisease,
            CASE WHEN HeartDisease = 'Yes' THEN 1 ELSE 0 END AS HeartDisease_Binary,

            -- Demographics
            Sex,
            AgeCategory,
            -- Extract numeric age from category (midpoint)
            CASE
                WHEN AgeCategory = '18-24' THEN 21
                WHEN AgeCategory = '25-29' THEN 27
                WHEN AgeCategory = '30-34' THEN 32
                WHEN AgeCategory = '35-39' THEN 37
                WHEN AgeCategory = '40-44' THEN 42
                WHEN AgeCategory = '45-49' THEN 47
                WHEN AgeCategory = '50-54' THEN 52
                WHEN AgeCategory = '55-59' THEN 57
                WHEN AgeCategory = '60-64' THEN 62
                WHEN AgeCategory = '65-69' THEN 67
                WHEN AgeCategory = '70-74' THEN 72
                WHEN AgeCategory = '75-79' THEN 77
                WHEN AgeCategory = '80 or older' THEN 82
                ELSE NULL
            END AS Age_Numeric,
            -- Broader age grouping
            CASE
                WHEN AgeCategory IN ('18-24', '25-29') THEN 'Young (18-29)'
                WHEN AgeCategory IN ('30-34', '35-39', '40-44') THEN 'Middle (30-44)'
                WHEN AgeCategory IN ('45-49', '50-54', '55-59') THEN 'Senior (45-59)'
                ELSE 'Elderly (60+)'
            END AS Age_Group,
            Race,

            -- Health metrics (type casting with validation)
            CAST(
                CASE
                    WHEN TRY_CAST(BMI AS DECIMAL(5,2)) BETWEEN 10.0 AND 100.0
                    THEN BMI ELSE NULL
                END AS DECIMAL(5,2)
            ) AS BMI,

            -- BMI Category
            CASE
                WHEN TRY_CAST(BMI AS DECIMAL(5,2)) < 18.5 THEN 'Underweight'
                WHEN TRY_CAST(BMI AS DECIMAL(5,2)) < 25.0 THEN 'Normal'
                WHEN TRY_CAST(BMI AS DECIMAL(5,2)) < 30.0 THEN 'Overweight'
                WHEN TRY_CAST(BMI AS DECIMAL(5,2)) >= 30.0 THEN 'Obese'
                ELSE 'Unknown'
            END AS BMI_Category,

            GenHealth,

            -- Health Score (composite 0-100)
            CAST(
                ROUND(
                    (
                        -- GenHealth component (0-40 points)
                        CASE GenHealth
                            WHEN 'Excellent' THEN 40
                            WHEN 'Very good' THEN 32
                            WHEN 'Good' THEN 24
                            WHEN 'Fair' THEN 12
                            WHEN 'Poor' THEN 0
                            ELSE 20
                        END
                        -- Physical health component (0-30 points, inverted)
                        + (30.0 - ISNULL(TRY_CAST(PhysicalHealth AS DECIMAL(5,2)), 0))
                        -- Mental health component (0-30 points, inverted)
                        + (30.0 - ISNULL(TRY_CAST(MentalHealth AS DECIMAL(5,2)), 0))
                    ), 2
                )
            AS DECIMAL(5,2)) AS Health_Score,

            CAST(TRY_CAST(PhysicalHealth AS FLOAT) AS TINYINT) AS PhysicalHealth,
            CAST(TRY_CAST(MentalHealth AS FLOAT) AS TINYINT) AS MentalHealth,

            CAST(TRY_CAST(SleepTime AS FLOAT) AS TINYINT) AS SleepTime,
            -- Sleep Quality
            CASE
                WHEN TRY_CAST(SleepTime AS FLOAT) < 5 THEN 'Poor'
                WHEN TRY_CAST(SleepTime AS FLOAT) < 7 THEN 'Fair'
                WHEN TRY_CAST(SleepTime AS FLOAT) <= 9 THEN 'Good'
                WHEN TRY_CAST(SleepTime AS FLOAT) > 9 THEN 'Excessive'
                ELSE 'Unknown'
            END AS Sleep_Quality,

            DiffWalking,

            -- Lifestyle
            Smoking,
            AlcoholDrinking,
            PhysicalActivity,

            -- Conditions
            Stroke,
            Diabetic,
            Asthma,
            KidneyDisease,
            SkinCancer,

            -- Comorbidity Count
            CAST(
                (CASE WHEN Diabetic IN ('Yes', 'Yes (during pregnancy)') THEN 1 ELSE 0 END) +
                (CASE WHEN Asthma = 'Yes' THEN 1 ELSE 0 END) +
                (CASE WHEN KidneyDisease = 'Yes' THEN 1 ELSE 0 END) +
                (CASE WHEN SkinCancer = 'Yes' THEN 1 ELSE 0 END) +
                (CASE WHEN Stroke = 'Yes' THEN 1 ELSE 0 END)
            AS TINYINT) AS Comorbidity_Count,

            -- Comorbidity Level
            CASE
                WHEN (CASE WHEN Diabetic IN ('Yes', 'Yes (during pregnancy)') THEN 1 ELSE 0 END) +
                     (CASE WHEN Asthma = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN KidneyDisease = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN SkinCancer = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN Stroke = 'Yes' THEN 1 ELSE 0 END) = 0 THEN 'None'
                WHEN (CASE WHEN Diabetic IN ('Yes', 'Yes (during pregnancy)') THEN 1 ELSE 0 END) +
                     (CASE WHEN Asthma = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN KidneyDisease = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN SkinCancer = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN Stroke = 'Yes' THEN 1 ELSE 0 END) = 1 THEN 'Low'
                WHEN (CASE WHEN Diabetic IN ('Yes', 'Yes (during pregnancy)') THEN 1 ELSE 0 END) +
                     (CASE WHEN Asthma = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN KidneyDisease = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN SkinCancer = 'Yes' THEN 1 ELSE 0 END) +
                     (CASE WHEN Stroke = 'Yes' THEN 1 ELSE 0 END) <= 3 THEN 'Moderate'
                ELSE 'High'
            END AS Comorbidity_Level,

            '{batch_id}' AS LoadBatchID

        FROM bronze.PatientData
        WHERE LoadBatchID = '{batch_id}'
          AND TRY_CAST(BMI AS FLOAT) BETWEEN 10.0 AND 100.0
          AND TRY_CAST(SleepTime AS FLOAT) BETWEEN 1 AND 24
          AND TRY_CAST(PhysicalHealth AS FLOAT) BETWEEN 0 AND 30
          AND TRY_CAST(MentalHealth AS FLOAT) BETWEEN 0 AND 30
    """

    with get_sql_connection() as conn:
        cursor = conn.cursor()

        # Count bronze rows for this batch
        cursor.execute(f"SELECT COUNT(*) FROM bronze.PatientData WHERE LoadBatchID = '{batch_id}'")
        bronze_count = cursor.fetchone()[0]

        if bronze_count == 0:
            # Fallback: use all bronze rows if batch doesn't match
            cursor.execute("SELECT COUNT(*) FROM bronze.PatientData")
            bronze_count = cursor.fetchone()[0]
            # Update the SQL to not filter by batch
            transform_sql = transform_sql.replace(
                f"WHERE LoadBatchID = '{batch_id}'",
                "WHERE 1=1"
            ).replace(
                f"'{batch_id}' AS LoadBatchID",
                f"'{batch_id}' AS LoadBatchID"
            )

        print(f"🧹 Cleaning {bronze_count:,} rows from Bronze → Silver...")

        # Truncate silver before loading
        cursor.execute("TRUNCATE TABLE silver.CleanedPatientData")
        conn.commit()

        # Execute transformation
        cursor.execute(transform_sql)
        inserted = cursor.rowcount
        rejected = bronze_count - inserted
        conn.commit()
        cursor.close()

    end = datetime.now()
    duration = (end - start).total_seconds()

    print(f"✅ Silver load complete: {inserted:,} inserted, {rejected:,} rejected in {duration:.1f}s")

    _log_step(batch_id, "silver", "Bronze → Silver", bronze_count, inserted, rejected, start, end, "SUCCESS")

    result.update({
        "status": "SUCCESS",
        "rows_processed": bronze_count,
        "rows_inserted": inserted,
        "rows_rejected": rejected,
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
    load_silver(bid)
