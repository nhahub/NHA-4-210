-- ============================================================
-- 02_StagingLayer.sql
-- Staging Layer — Raw Landing Zone (1:1 copy from CSV)
-- All columns are VARCHAR to accept any raw data without rejection
-- ============================================================

USE HeartDiseaseDB;
GO

-- ────────────────────────────────────────────────────────────
-- Drop and recreate staging table (truncated on every load)
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('stg.RawPatientData', 'U') IS NOT NULL
    DROP TABLE stg.RawPatientData;
GO

CREATE TABLE stg.RawPatientData (
    StagingID           INT IDENTITY(1,1)   PRIMARY KEY,
    HeartDisease        VARCHAR(50)         NULL,
    BMI                 VARCHAR(50)         NULL,
    Smoking             VARCHAR(50)         NULL,
    AlcoholDrinking     VARCHAR(50)         NULL,
    Stroke              VARCHAR(50)         NULL,
    PhysicalHealth      VARCHAR(50)         NULL,
    MentalHealth        VARCHAR(50)         NULL,
    DiffWalking         VARCHAR(50)         NULL,
    Sex                 VARCHAR(50)         NULL,
    AgeCategory         VARCHAR(50)         NULL,
    Race                VARCHAR(100)        NULL,
    Diabetic            VARCHAR(100)        NULL,
    PhysicalActivity    VARCHAR(50)         NULL,
    GenHealth           VARCHAR(50)         NULL,
    SleepTime           VARCHAR(50)         NULL,
    Asthma              VARCHAR(50)         NULL,
    KidneyDisease       VARCHAR(50)         NULL,
    SkinCancer          VARCHAR(50)         NULL
);
GO

PRINT '✅ Staging table [stg.RawPatientData] created successfully.';
GO
