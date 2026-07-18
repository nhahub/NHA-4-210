-- ============================================================
-- 01_CreateDatabase.sql
-- Medallion Architecture — Schema Setup
-- Adds layered schemas to the existing HeartDiseaseDB
-- ============================================================

USE HeartDiseaseDB;
GO

-- ────────────────────────────────────────────────────────────
-- Create Medallion Schemas (if they don't already exist)
-- ────────────────────────────────────────────────────────────

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'stg')
    EXEC('CREATE SCHEMA stg');
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'bronze')
    EXEC('CREATE SCHEMA bronze');
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'silver')
    EXEC('CREATE SCHEMA silver');
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'gold')
    EXEC('CREATE SCHEMA gold');
GO

IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'ops')
    EXEC('CREATE SCHEMA ops');
GO

PRINT '✅ All Medallion schemas created successfully.';
GO
