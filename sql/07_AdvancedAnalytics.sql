-- ============================================================
-- 07_AdvancedAnalytics.sql
-- Advanced SQL Analytics — CTEs, NTILE, Stored Procedures
-- ============================================================

USE HeartDiseaseDB;
GO

-- ============================================================
-- Q16: Multi-Factor Risk Analysis using CTE
-- حساب عدد عوامل الخطر لكل مريض وتصنيفهم
-- ============================================================
WITH RiskFactors AS (
    SELECT
        f.PatientID,
        f.HeartDisease_Binary,
        (CASE WHEN l.Smoking = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN c.Diabetic_Binary = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN h.BMI_Category IN ('Obese', 'Overweight') THEN 1 ELSE 0 END +
         CASE WHEN l.PhysicalActivity = 'No' THEN 1 ELSE 0 END +
         CASE WHEN c.Stroke = 'Yes' THEN 1 ELSE 0 END +
         CASE WHEN c.KidneyDisease = 'Yes' THEN 1 ELSE 0 END
        ) AS Risk_Factor_Count
    FROM gold.vw_fact_Diagnosis f
    INNER JOIN gold.vw_dim_Lifestyle l ON f.PatientID = l.PatientID
    INNER JOIN gold.vw_dim_Conditions c ON f.PatientID = c.PatientID
    INNER JOIN gold.vw_dim_HealthStatus h ON f.PatientID = h.PatientID
)
SELECT
    Risk_Factor_Count,
    CASE
        WHEN Risk_Factor_Count = 0 THEN 'No Risk Factors'
        WHEN Risk_Factor_Count <= 2 THEN 'Low Risk (1-2)'
        WHEN Risk_Factor_Count <= 4 THEN 'Moderate Risk (3-4)'
        ELSE 'High Risk (5+)'
    END AS Risk_Category,
    COUNT(*) AS Total_Patients,
    SUM(HeartDisease_Binary) AS Has_Disease,
    ROUND(SUM(HeartDisease_Binary) * 100.0 / COUNT(*), 1) AS Disease_Rate_Pct
FROM RiskFactors
GROUP BY Risk_Factor_Count
ORDER BY Risk_Factor_Count;
GO

-- ============================================================
-- Q17: ML Model Validation — Confusion Matrix from SQL
-- مقارنة توقعات نموذج ML مع التشخيص الفعلي
-- ============================================================
WITH ModelValidation AS (
    SELECT
        HeartDisease_Binary AS Actual,
        ML_Prediction AS Predicted,
        COUNT(*) AS Cnt
    FROM fact_Diagnosis
    GROUP BY HeartDisease_Binary, ML_Prediction
)
SELECT
    CASE
        WHEN Actual = 1 AND Predicted = 1 THEN 'True Positive (TP)'
        WHEN Actual = 0 AND Predicted = 0 THEN 'True Negative (TN)'
        WHEN Actual = 0 AND Predicted = 1 THEN 'False Positive (FP)'
        WHEN Actual = 1 AND Predicted = 0 THEN 'False Negative (FN)'
    END AS Classification,
    Cnt AS Patient_Count,
    ROUND(Cnt * 100.0 / SUM(Cnt) OVER(), 2) AS Percentage
FROM ModelValidation
ORDER BY Actual DESC, Predicted DESC;
GO

-- ============================================================
-- Q18: Geographic Quartile Analysis using NTILE
-- تقسيم الولايات إلى أرباع حسب معدل انتشار أمراض القلب
-- ============================================================
SELECT
    State,
    Region,
    HD_Prevalence_Pct,
    Obesity_Rate_Pct,
    Smoking_Rate_Pct,
    NTILE(4) OVER (ORDER BY HD_Prevalence_Pct DESC) AS Risk_Quartile,
    CASE NTILE(4) OVER (ORDER BY HD_Prevalence_Pct DESC)
        WHEN 1 THEN 'Q1 — Highest Risk'
        WHEN 2 THEN 'Q2 — Above Average'
        WHEN 3 THEN 'Q3 — Below Average'
        WHEN 4 THEN 'Q4 — Lowest Risk'
    END AS Quartile_Label
FROM gold.vw_dim_Geography
ORDER BY HD_Prevalence_Pct DESC;
GO

-- ============================================================
-- Q19: Cross-Tab Analysis — Smoking × Diabetes × Obesity
-- مصفوفة عوامل الخطر المتقاطعة
-- ============================================================
SELECT
    l.Smoking,
    c.Diabetic_Binary AS Diabetic,
    h.BMI_Category,
    COUNT(*) AS Total_Patients,
    SUM(f.HeartDisease_Binary) AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1) AS Disease_Rate_Pct,
    ROUND(AVG(f.Risk_Score), 2) AS Avg_Risk_Score
FROM gold.vw_fact_Diagnosis f
INNER JOIN gold.vw_dim_Lifestyle l ON f.PatientID = l.PatientID
INNER JOIN gold.vw_dim_Conditions c ON f.PatientID = c.PatientID
INNER JOIN gold.vw_dim_HealthStatus h ON f.PatientID = h.PatientID
WHERE h.BMI_Category IN ('Normal', 'Obese')
GROUP BY l.Smoking, c.Diabetic_Binary, h.BMI_Category
HAVING COUNT(*) > 100
ORDER BY Disease_Rate_Pct DESC;
GO

-- ============================================================
-- Q20: Stored Procedure — Patient Risk Report by Age Group
-- إجراء مخزن يعيد تقريراً شاملاً لفئة عمرية محددة
-- ============================================================
IF OBJECT_ID('dbo.sp_PatientRiskReport', 'P') IS NOT NULL
    DROP PROCEDURE dbo.sp_PatientRiskReport;
GO

CREATE PROCEDURE dbo.sp_PatientRiskReport
    @AgeGroup NVARCHAR(20) = NULL
AS
BEGIN
    SET NOCOUNT ON;

    -- Section 1: Overview Statistics
    SELECT
        'Overview' AS Section,
        p.Age_Group,
        COUNT(*) AS Total_Patients,
        SUM(f.HeartDisease_Binary) AS Heart_Disease_Count,
        ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1) AS Disease_Rate_Pct,
        ROUND(AVG(f.Risk_Score), 2) AS Avg_Risk_Score,
        ROUND(AVG(f.ML_Probability), 3) AS Avg_ML_Probability
    FROM fact_Diagnosis f
    INNER JOIN dim_Patients p ON f.PatientID = p.PatientID
    WHERE (@AgeGroup IS NULL OR p.Age_Group = @AgeGroup)
    GROUP BY p.Age_Group
    ORDER BY p.Age_Group;

    -- Section 2: Risk Tier Distribution
    SELECT
        'Risk Distribution' AS Section,
        f.Risk_Tier_3Levels,
        COUNT(*) AS Patient_Count,
        ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS Pct
    FROM fact_Diagnosis f
    INNER JOIN dim_Patients p ON f.PatientID = p.PatientID
    WHERE (@AgeGroup IS NULL OR p.Age_Group = @AgeGroup)
    GROUP BY f.Risk_Tier_3Levels
    ORDER BY
        CASE f.Risk_Tier_3Levels
            WHEN 'High' THEN 1
            WHEN 'Medium' THEN 2
            WHEN 'Low' THEN 3
        END;

    -- Section 3: Top Risk Factors
    SELECT
        'Top Risk Factors' AS Section,
        'Smoking' AS Factor,
        ROUND(SUM(CASE WHEN l.Smoking = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1) AS Prevalence_Pct
    FROM gold.vw_fact_Diagnosis f
    INNER JOIN gold.vw_dim_Patients p ON f.PatientID = p.PatientID
    INNER JOIN gold.vw_dim_Lifestyle l ON f.PatientID = l.PatientID
    WHERE (@AgeGroup IS NULL OR p.Age_Group = @AgeGroup)
      AND f.HeartDisease_Binary = 1
    UNION ALL
    SELECT
        'Top Risk Factors', 'Obesity',
        ROUND(SUM(CASE WHEN h.BMI_Category = 'Obese' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)
    FROM gold.vw_fact_Diagnosis f
    INNER JOIN gold.vw_dim_Patients p ON f.PatientID = p.PatientID
    INNER JOIN gold.vw_dim_HealthStatus h ON f.PatientID = h.PatientID
    WHERE (@AgeGroup IS NULL OR p.Age_Group = @AgeGroup)
      AND f.HeartDisease_Binary = 1
    UNION ALL
    SELECT
        'Top Risk Factors', 'Diabetes',
        ROUND(SUM(CASE WHEN c.Diabetic_Binary = 'Yes' THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 1)
    FROM gold.vw_fact_Diagnosis f
    INNER JOIN gold.vw_dim_Patients p ON f.PatientID = p.PatientID
    INNER JOIN gold.vw_dim_Conditions c ON f.PatientID = c.PatientID
    WHERE (@AgeGroup IS NULL OR p.Age_Group = @AgeGroup)
      AND f.HeartDisease_Binary = 1
    ORDER BY Prevalence_Pct DESC;
END;
GO

-- ============================================================
-- Q21: Running Total — Cumulative Disease Count by Age
-- التراكمي التصاعدي لعدد المرضى حسب الفئة العمرية
-- ============================================================
SELECT
    p.AgeCategory,
    COUNT(*) AS Total_Patients,
    SUM(f.HeartDisease_Binary) AS Has_Disease,
    SUM(SUM(f.HeartDisease_Binary)) OVER (ORDER BY p.AgeCategory
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS Cumulative_Disease_Count,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1) AS Disease_Rate_Pct
FROM gold.vw_fact_Diagnosis f
INNER JOIN gold.vw_dim_Patients p ON f.PatientID = p.PatientID
GROUP BY p.AgeCategory
ORDER BY p.AgeCategory;
GO

PRINT '✅ Advanced analytics queries and stored procedure created successfully.';
GO
