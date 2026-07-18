-- ============================================================
-- 03_BronzeLayer.sql
-- Bronze Layer — Durable Historical Copy with Audit Metadata
-- Preserves every load batch for traceability
-- ============================================================

USE HeartDiseaseDB;
GO

-- ────────────────────────────────────────────────────────────
-- Bronze table (append-only, never truncated)
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('bronze.PatientData', 'U') IS NOT NULL
    DROP TABLE bronze.PatientData;
GO

CREATE TABLE bronze.PatientData (
    BronzeID            INT IDENTITY(1,1)       PRIMARY KEY,

    -- ── Original raw columns (same as staging) ─────────────
    HeartDisease        VARCHAR(50)             NULL,
    BMI                 VARCHAR(50)             NULL,
    Smoking             VARCHAR(50)             NULL,
    AlcoholDrinking     VARCHAR(50)             NULL,
    Stroke              VARCHAR(50)             NULL,
    PhysicalHealth      VARCHAR(50)             NULL,
    MentalHealth        VARCHAR(50)             NULL,
    DiffWalking         VARCHAR(50)             NULL,
    Sex                 VARCHAR(50)             NULL,
    AgeCategory         VARCHAR(50)             NULL,
    Race                VARCHAR(100)            NULL,
    Diabetic            VARCHAR(100)            NULL,
    PhysicalActivity    VARCHAR(50)             NULL,
    GenHealth           VARCHAR(50)             NULL,
    SleepTime           VARCHAR(50)             NULL,
    Asthma              VARCHAR(50)             NULL,
    KidneyDisease       VARCHAR(50)             NULL,
    SkinCancer          VARCHAR(50)             NULL,

    -- ── Audit metadata columns ─────────────────────────────
    LoadBatchID         UNIQUEIDENTIFIER        NOT NULL,
    LoadTimestamp       DATETIME2               NOT NULL DEFAULT SYSDATETIME(),
    RecordHash          VARCHAR(64)             NOT NULL,
    RecordSource        VARCHAR(255)            NOT NULL DEFAULT 'heart_2020_cleaned.csv'
);
GO

-- Index on batch for fast audit queries
CREATE NONCLUSTERED INDEX IX_bronze_BatchID
    ON bronze.PatientData (LoadBatchID);
GO

-- Index on hash for deduplication checks
CREATE NONCLUSTERED INDEX IX_bronze_RecordHash
    ON bronze.PatientData (RecordHash);
GO

PRINT '✅ Bronze table [bronze.PatientData] created successfully.';
GO
