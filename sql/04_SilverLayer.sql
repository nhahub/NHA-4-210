-- ============================================================
-- 04_SilverLayer.sql
-- Silver Layer — Cleaned, Validated, and Properly Typed Data
-- Business rules applied, derived columns computed
-- ============================================================

USE HeartDiseaseDB;
GO

-- ────────────────────────────────────────────────────────────
-- Silver table (clean, typed, validated data)
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('silver.CleanedPatientData', 'U') IS NOT NULL
    DROP TABLE silver.CleanedPatientData;
GO

CREATE TABLE silver.CleanedPatientData (
    SilverID                INT IDENTITY(1,1)       PRIMARY KEY,
    PatientID               VARCHAR(10)             NOT NULL,

    -- ── Core diagnosis fields ──────────────────────────────
    HeartDisease            NVARCHAR(5)             NOT NULL,
    HeartDisease_Binary     TINYINT                 NOT NULL,

    -- ── Demographics ───────────────────────────────────────
    Sex                     NVARCHAR(10)            NOT NULL,
    AgeCategory             NVARCHAR(15)            NOT NULL,
    Age_Numeric             TINYINT                 NULL,
    Age_Group               NVARCHAR(20)            NULL,
    Race                    NVARCHAR(35)            NOT NULL,

    -- ── Health metrics ─────────────────────────────────────
    BMI                     DECIMAL(5,2)            NOT NULL,
    BMI_Category            NVARCHAR(20)            NOT NULL,
    GenHealth               NVARCHAR(15)            NOT NULL,
    Health_Score            DECIMAL(5,2)            NULL,
    PhysicalHealth          TINYINT                 NOT NULL,
    MentalHealth            TINYINT                 NOT NULL,
    SleepTime               TINYINT                 NOT NULL,
    Sleep_Quality           NVARCHAR(12)            NULL,
    DiffWalking             NVARCHAR(5)             NOT NULL,

    -- ── Lifestyle ──────────────────────────────────────────
    Smoking                 NVARCHAR(5)             NOT NULL,
    AlcoholDrinking         NVARCHAR(5)             NOT NULL,
    PhysicalActivity        NVARCHAR(5)             NOT NULL,

    -- ── Conditions ─────────────────────────────────────────
    Stroke                  NVARCHAR(5)             NOT NULL,
    Diabetic                NVARCHAR(35)            NOT NULL,
    Asthma                  NVARCHAR(5)             NOT NULL,
    KidneyDisease           NVARCHAR(5)             NOT NULL,
    SkinCancer              NVARCHAR(5)             NOT NULL,
    Comorbidity_Count       TINYINT                 NULL,
    Comorbidity_Level       NVARCHAR(12)            NULL,

    -- ── Audit ──────────────────────────────────────────────
    LoadBatchID             UNIQUEIDENTIFIER        NOT NULL,
    CleanedTimestamp        DATETIME2               NOT NULL DEFAULT SYSDATETIME(),

    -- ── Constraints ────────────────────────────────────────
    CONSTRAINT CK_Silver_BMI CHECK (BMI BETWEEN 10.0 AND 100.0),
    CONSTRAINT CK_Silver_Sleep CHECK (SleepTime BETWEEN 1 AND 24),
    CONSTRAINT CK_Silver_PhysHealth CHECK (PhysicalHealth BETWEEN 0 AND 30),
    CONSTRAINT CK_Silver_MentHealth CHECK (MentalHealth BETWEEN 0 AND 30),
    CONSTRAINT CK_Silver_HeartBin CHECK (HeartDisease_Binary IN (0, 1))
);
GO

-- Index on PatientID for Gold View joins
CREATE NONCLUSTERED INDEX IX_silver_PatientID
    ON silver.CleanedPatientData (PatientID);
GO

PRINT '✅ Silver table [silver.CleanedPatientData] created successfully.';
GO
