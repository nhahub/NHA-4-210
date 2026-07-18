
USE HeartDiseaseDB;
GO
-- ============================================================
-- Q1: كم عدد المرضى المصابين بأمراض القلب مقارنة بالأصحاء؟
-- ============================================================
SELECT
    HeartDisease        AS Disease_Status,
    COUNT(*)            AS Patient_Count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 1) AS Percentage
FROM fact_Diagnosis
GROUP BY HeartDisease
ORDER BY Patient_Count DESC;

-- ============================================================
-- Q2: ما هو توزيع مرضى القلب حسب الجنس؟
-- ============================================================
SELECT
    p.Sex,
    COUNT(*)                               AS Total_Patients,
    SUM(f.HeartDisease_Binary)             AS Has_Disease,
    COUNT(*) - SUM(f.HeartDisease_Binary)  AS No_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1) AS Disease_Rate_Pct
FROM fact_Diagnosis f
INNER JOIN dim_Patients p ON f.PatientID = p.PatientID
GROUP BY p.Sex
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q3: ما هو توزيع مرضى القلب حسب الفئة العمرية؟
-- ============================================================
SELECT
    p.Age_Group,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct
FROM fact_Diagnosis f
INNER JOIN dim_Patients p ON f.PatientID = p.PatientID
GROUP BY p.Age_Group
ORDER BY p.Age_Group;

-- ============================================================
-- Q4: ما هو تأثير التدخين على الإصابة بأمراض القلب؟
-- ============================================================
SELECT
    l.Smoking,
    COUNT(*)                                                 AS Total_Patients,
    SUM(f.HeartDisease_Binary)                               AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct
FROM fact_Diagnosis f
INNER JOIN dim_Lifestyle l ON f.PatientID = l.PatientID
GROUP BY l.Smoking
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q5: كيف يؤثر مؤشر كتلة الجسم (BMI) على الإصابة بأمراض القلب؟
-- ============================================================
SELECT
    h.BMI_Category,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct,
    ROUND(AVG(h.BMI), 1)                                      AS Avg_BMI
FROM fact_Diagnosis f
INNER JOIN dim_HealthStatus h ON f.PatientID = h.PatientID
GROUP BY h.BMI_Category
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q6: ما هي العلاقة بين مرض السكري وأمراض القلب؟
-- ============================================================
SELECT
    c.Diabetic,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct
FROM fact_Diagnosis f
INNER JOIN dim_Conditions c ON f.PatientID = c.PatientID
GROUP BY c.Diabetic
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q7: ما هو تأثير الحالة الصحية العامة (GenHealth) على مرض القلب؟
-- ============================================================
SELECT
    h.GenHealth,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct,
    ROUND(AVG(h.Health_Score), 2)                             AS Avg_Health_Score
FROM fact_Diagnosis f
INNER JOIN dim_HealthStatus h ON f.PatientID = h.PatientID
GROUP BY h.GenHealth
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q8: ما هي أعلى 10 ولايات في نسبة انتشار أمراض القلب؟
-- ============================================================
SELECT TOP 10
    g.State,
    g.Region,
    g.HD_Prevalence_Pct,
    g.Obesity_Rate_Pct,
    g.Smoking_Rate_Pct,
    g.HD_Risk_Label
FROM dim_Geography g
ORDER BY g.HD_Prevalence_Pct DESC;

-- ============================================================
-- Q9: ما هو توزيع المرضى حسب مستوى الخطر (Risk Tier)؟
-- ============================================================
SELECT
    Risk_Tier_3Levels,
    COUNT(*)                                            AS Total_Patients,
    SUM(HeartDisease_Binary)                            AS Has_Disease,
    ROUND(AVG(Risk_Score), 2)                           AS Avg_Risk_Score,
    ROUND(SUM(HeartDisease_Binary) * 100.0 / COUNT(*), 1) AS Disease_Rate_Pct
FROM fact_Diagnosis
GROUP BY Risk_Tier_3Levels
ORDER BY Avg_Risk_Score DESC;

-- ============================================================
-- Q10: كيف يؤثر النشاط البدني على الإصابة بأمراض القلب؟
-- ============================================================
SELECT
    l.PhysicalActivity,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct
FROM fact_Diagnosis f
INNER JOIN dim_Lifestyle l ON f.PatientID = l.PatientID
GROUP BY l.PhysicalActivity
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q11: ما هو تأثير جودة النوم على أمراض القلب؟
-- ============================================================
SELECT
    h.Sleep_Quality,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct,
    ROUND(AVG(CAST(h.SleepTime AS DECIMAL(5,2))), 1)          AS Avg_Sleep_Hours
FROM fact_Diagnosis f
INNER JOIN dim_HealthStatus h ON f.PatientID = h.PatientID
GROUP BY h.Sleep_Quality
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q12: ما هو عدد المرضى الحرجين (Is_Critical) وتوزيعهم؟
-- ============================================================
SELECT
    CASE WHEN Is_Critical = 1 THEN 'Critical' ELSE 'Not Critical' END AS Critical_Status,
    COUNT(*)                  AS Total_Patients,
    SUM(HeartDisease_Binary)  AS Has_Disease,
    ROUND(AVG(Risk_Score), 2) AS Avg_Risk_Score,
    ROUND(AVG(ML_Probability), 3) AS Avg_ML_Probability
FROM fact_Diagnosis
GROUP BY Is_Critical;

-- ============================================================
-- Q13: ما هو تأثير تراكم الأمراض المصاحبة على مرض القلب؟
-- ============================================================
SELECT
    c.Comorbidity_Level,
    c.Comorbidity_Count,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct
FROM fact_Diagnosis f
INNER JOIN dim_Conditions c ON f.PatientID = c.PatientID
GROUP BY c.Comorbidity_Level, c.Comorbidity_Count
ORDER BY c.Comorbidity_Count;

-- ============================================================
-- Q14: ما هو توزيع مرضى القلب حسب العرق (Race)؟
-- ============================================================
SELECT
    p.Race,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct
FROM fact_Diagnosis f
INNER JOIN dim_Patients p ON f.PatientID = p.PatientID
GROUP BY p.Race
ORDER BY Disease_Rate_Pct DESC;

-- ============================================================
-- Q15: ما هي أبرز خصائص المرضى في كل شريحة (Patient Segment)؟
-- ============================================================
SELECT
    f.Patient_Segment_Label,
    COUNT(*)                                                  AS Total_Patients,
    SUM(f.HeartDisease_Binary)                                AS Has_Disease,
    ROUND(SUM(f.HeartDisease_Binary) * 100.0 / COUNT(*), 1)  AS Disease_Rate_Pct,
    ROUND(AVG(f.Risk_Score), 2)                               AS Avg_Risk_Score,
    ROUND(AVG(h.BMI), 1)                                      AS Avg_BMI,
    ROUND(AVG(CAST(h.SleepTime AS DECIMAL(5,2))), 1)          AS Avg_Sleep_Hours,
    ROUND(AVG(h.Health_Score), 2)                             AS Avg_Health_Score
FROM fact_Diagnosis f
INNER JOIN dim_HealthStatus h ON f.PatientID = h.PatientID
GROUP BY f.Patient_Segment_Label
ORDER BY Disease_Rate_Pct DESC;
