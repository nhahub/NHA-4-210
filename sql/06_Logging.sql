-- ============================================================
-- 06_Logging.sql
-- Pipeline Observability — ETL Logging Table & Helper SP
-- ============================================================

USE HeartDiseaseDB;
GO

-- ────────────────────────────────────────────────────────────
-- Pipeline Log Table
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('ops.PipelineLog', 'U') IS NOT NULL
    DROP TABLE ops.PipelineLog;
GO

CREATE TABLE ops.PipelineLog (
    LogID               INT IDENTITY(1,1) PRIMARY KEY,
    BatchID             UNIQUEIDENTIFIER    NOT NULL,
    LayerName           VARCHAR(20)         NOT NULL,   -- staging, bronze, silver, gold
    StepName            VARCHAR(100)        NOT NULL,
    RowsProcessed       INT                 NULL DEFAULT 0,
    RowsInserted        INT                 NULL DEFAULT 0,
    RowsRejected        INT                 NULL DEFAULT 0,
    StartTime           DATETIME2           NOT NULL,
    EndTime             DATETIME2           NULL,
    DurationSeconds     AS DATEDIFF(SECOND, StartTime, EndTime),
    Status              VARCHAR(20)         NOT NULL DEFAULT 'RUNNING',  -- RUNNING, SUCCESS, FAILED
    ErrorMessage        NVARCHAR(2000)      NULL,
    CreatedAt           DATETIME2           NOT NULL DEFAULT SYSDATETIME()
);
GO

-- ────────────────────────────────────────────────────────────
-- Helper Stored Procedure: Log a Pipeline Step
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('ops.sp_LogStep', 'P') IS NOT NULL
    DROP PROCEDURE ops.sp_LogStep;
GO

CREATE PROCEDURE ops.sp_LogStep
    @BatchID        UNIQUEIDENTIFIER,
    @LayerName      VARCHAR(20),
    @StepName       VARCHAR(100),
    @RowsProcessed  INT = 0,
    @RowsInserted   INT = 0,
    @RowsRejected   INT = 0,
    @Status         VARCHAR(20) = 'SUCCESS',
    @ErrorMessage   NVARCHAR(2000) = NULL,
    @StartTime      DATETIME2 = NULL,
    @EndTime        DATETIME2 = NULL
AS
BEGIN
    SET NOCOUNT ON;

    INSERT INTO ops.PipelineLog (
        BatchID, LayerName, StepName,
        RowsProcessed, RowsInserted, RowsRejected,
        StartTime, EndTime, Status, ErrorMessage
    )
    VALUES (
        @BatchID, @LayerName, @StepName,
        @RowsProcessed, @RowsInserted, @RowsRejected,
        ISNULL(@StartTime, SYSDATETIME()),
        ISNULL(@EndTime, SYSDATETIME()),
        @Status, @ErrorMessage
    );
END;
GO

-- ────────────────────────────────────────────────────────────
-- View: Latest Pipeline Run Summary
-- ────────────────────────────────────────────────────────────
IF OBJECT_ID('ops.vw_LatestPipelineRun', 'V') IS NOT NULL
    DROP VIEW ops.vw_LatestPipelineRun;
GO

CREATE VIEW ops.vw_LatestPipelineRun AS
WITH LatestBatch AS (
    SELECT TOP 1 BatchID
    FROM ops.PipelineLog
    ORDER BY CreatedAt DESC
)
SELECT
    pl.LayerName,
    pl.StepName,
    pl.RowsProcessed,
    pl.RowsInserted,
    pl.RowsRejected,
    pl.DurationSeconds,
    pl.Status,
    pl.ErrorMessage,
    pl.StartTime,
    pl.EndTime
FROM ops.PipelineLog pl
INNER JOIN LatestBatch lb ON pl.BatchID = lb.BatchID
GO

PRINT '✅ Logging table, stored procedure, and summary view created successfully.';
GO
