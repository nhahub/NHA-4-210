-- ============================================================
-- 05_GoldLayer.sql
-- Gold Layer — Business-Ready Star Schema Views
-- These views sit on top of Silver and match the existing
-- dbo dimension/fact structure for Power BI compatibility
-- ============================================================

USE HeartDiseaseDB;
GO

-- ────────────────────────────────────────────────────────────
-- gold.vw_dim_Patients
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('gold.vw_dim_Patients', 'V') IS NOT NULL
    DROP VIEW gold.vw_dim_Patients;
GO

CREATE VIEW gold.vw_dim_Patients AS
SELECT DISTINCT
    PatientID,
    Sex,
    AgeCategory,
    Age_Numeric,
    Age_Group,
    Race
FROM silver.CleanedPatientData;
GO

-- ────────────────────────────────────────────────────────────
-- gold.vw_dim_Lifestyle
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('gold.vw_dim_Lifestyle', 'V') IS NOT NULL
    DROP VIEW gold.vw_dim_Lifestyle;
GO

CREATE VIEW gold.vw_dim_Lifestyle AS
SELECT DISTINCT
    PatientID,
    Smoking,
    AlcoholDrinking,
    PhysicalActivity,
    -- Lifestyle Score: count of healthy habits (0-3)
    CAST(
        CASE WHEN Smoking = 'No' THEN 1 ELSE 0 END +
        CASE WHEN AlcoholDrinking = 'No' THEN 1 ELSE 0 END +
        CASE WHEN PhysicalActivity = 'Yes' THEN 1 ELSE 0 END
    AS TINYINT) AS Lifestyle_Score,
    CASE
        WHEN (CASE WHEN Smoking = 'No' THEN 1 ELSE 0 END +
              CASE WHEN AlcoholDrinking = 'No' THEN 1 ELSE 0 END +
              CASE WHEN PhysicalActivity = 'Yes' THEN 1 ELSE 0 END) = 3 THEN 'Healthy'
        WHEN (CASE WHEN Smoking = 'No' THEN 1 ELSE 0 END +
              CASE WHEN AlcoholDrinking = 'No' THEN 1 ELSE 0 END +
              CASE WHEN PhysicalActivity = 'Yes' THEN 1 ELSE 0 END) >= 2 THEN 'Moderate'
        ELSE 'Unhealthy'
    END AS Lifestyle_Category
FROM silver.CleanedPatientData;
GO

-- ────────────────────────────────────────────────────────────
-- gold.vw_dim_HealthStatus
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('gold.vw_dim_HealthStatus', 'V') IS NOT NULL
    DROP VIEW gold.vw_dim_HealthStatus;
GO

CREATE VIEW gold.vw_dim_HealthStatus AS
SELECT DISTINCT
    PatientID,
    BMI,
    BMI_Category,
    CASE
        WHEN BMI < 18.5 THEN 'Low'
        WHEN BMI < 25.0 THEN 'Low'
        WHEN BMI < 30.0 THEN 'Medium'
        ELSE 'High'
    END AS BMI_Risk,
    GenHealth,
    Health_Score,
    PhysicalHealth,
    MentalHealth,
    SleepTime,
    Sleep_Quality,
    DiffWalking
FROM silver.CleanedPatientData;
GO

-- ────────────────────────────────────────────────────────────
-- gold.vw_dim_Conditions
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('gold.vw_dim_Conditions', 'V') IS NOT NULL
    DROP VIEW gold.vw_dim_Conditions;
GO

CREATE VIEW gold.vw_dim_Conditions AS
SELECT DISTINCT
    PatientID,
    Stroke,
    Diabetic,
    CASE
        WHEN Diabetic IN ('Yes', 'Yes (during pregnancy)') THEN 'Yes'
        ELSE 'No'
    END AS Diabetic_Binary,
    Asthma,
    KidneyDisease,
    SkinCancer,
    Comorbidity_Count,
    Comorbidity_Level
FROM silver.CleanedPatientData;
GO

-- ────────────────────────────────────────────────────────────
-- gold.vw_fact_Diagnosis
-- Central fact view joining Silver data with existing 
-- ML predictions and risk scores from dbo.fact_Diagnosis
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('gold.vw_fact_Diagnosis', 'V') IS NOT NULL
    DROP VIEW gold.vw_fact_Diagnosis;
GO

CREATE VIEW gold.vw_fact_Diagnosis AS
SELECT
    s.PatientID,
    s.HeartDisease,
    s.HeartDisease_Binary,

    -- Risk & ML columns from existing fact table
    ISNULL(f.Risk_Score, 0)                 AS Risk_Score,
    ISNULL(f.Risk_Tier_4Levels, 'Unknown')  AS Risk_Tier_4Levels,
    ISNULL(f.Risk_Tier_3Levels, 'Unknown')  AS Risk_Tier_3Levels,
    ISNULL(f.Is_Critical, 0)                AS Is_Critical,
    ISNULL(f.Is_High_Risk, 0)               AS Is_High_Risk,
    ISNULL(f.ML_Probability, 0)             AS ML_Probability,
    ISNULL(f.ML_Prediction, 0)              AS ML_Prediction,
    ISNULL(f.Patient_Segment, 0)            AS Patient_Segment,
    ISNULL(f.Patient_Segment_Label, 'Unknown') AS Patient_Segment_Label,

    -- FK references
    f.State,
    f.DateID
FROM silver.CleanedPatientData s
LEFT JOIN dbo.fact_Diagnosis f ON s.PatientID = f.PatientID;
GO

PRINT '✅ All Gold Layer views created successfully.';
GO
