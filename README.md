# 🫀 Heart Disease Diagnosis, Prediction & Analysis — DEPI Graduation Project 2026

> **مشروع التخرج لعام 2026 — نظام طبي ذكي متكامل للتنبؤ بمخاطر أمراض القلب وتحليل البيانات**
> 
> A comprehensive graduation project merging **Data Engineering (SQL, Star Schema)**, **Data Analysis (Excel, Power BI)**, and **Machine Learning & Explainable AI (Python, LightGBM, Streamlit, SHAP)** to predict and analyze heart disease risk using the CDC BRFSS dataset (319,795 patients).

---

## 🌐 Project Overview (نظرة عامة على المشروع)

This graduation project delivers an end-to-end medical decision support system. It is structured into four main components:
1.  📊 **Excel Data Analysis**: Initial data cleaning, pivot tables, and statistical profiling.
2.  🛢️ **SQL Database & Querying**: Multi-table Star Schema implementation, querying patient demographics, lifestyle habits, and risk metrics using SQL Server.
3.  📈 **Power BI Interactive Dashboard**: Professional dashboards visualizing key KPIs, patient segmentation, geographic risk distribution, and historical health metrics.
4.  🧠 **Machine Learning & Streamlit Web App**:
    *   Bayesian hyperparameter optimization using **Optuna** (LightGBM V2 model achieving **0.8404 ROC-AUC** and **80.57% Recall**).
    *   **Explainable AI (XAI)** using SHAP waterfall and beeswarm plots.
    *   An interactive **Streamlit web application** for real-time patient risk prediction and local explanation.

---

## 👥 Project Team (فريق العمل)

| Name | Role |
|------|------|
| **Abdelrahman Alnaggar** | Machine Learning & AI Pipeline + Power BI Dashboard |
| **Ahmed Elsayed** | Excel Data Analysis + SQL Database & Querying |
| **Ali Khaled** |

---

## 📂 Repository Structure (هيكل المجلدات)

```
Final Project/
├── README.md                      # This main project directory portal (English/Arabic)
├── Project_Documentation_AR.md     # Comprehensive academic & technical documentation in Arabic
├── .gitignore                     # Excludes temporary and extremely large files (like raw datasets)
│
├── Excel/                         # 1. Excel Component
│   └── DEPI Graduation Project.zip (Compressed archive of the 113.8MB Excel project; extract to view)
│
├── Sql/                           # 2. SQL Component
│   └── Depi Graduation Project SQL.sql # SQL Server analysis queries (demographics, habits, states)
│
├── power bi/                      # 3. Power BI Component
│   └── heart disease.pbix         # The Power BI interactive dashboard file (51.5 MB)
│
└── Machine Learning/              # 4. Machine Learning & Web App Component
    ├── app.py                     # Streamlit multi-version medical dashboard
    ├── heart_disease_project.ipynb # Jupyter Notebook: exploratory data analysis & training
    ├── Feature_Engineering_Heart_Disease.ipynb # Feature Engineering notebook (10 steps)
    ├── requirements.txt           # Python library dependencies
    ├── pipelines/                 # Training scripts
    │   ├── ml_pipeline.py         # Baseline V1 Pipeline (6 models)
    │   └── ml_pipeline_v2.py      # Tuned V2 Pipeline (Optuna optimization)
    └── outputs/                   # Visualizations, models, and metric logs
        ├── best_model.pkl         # V1 baseline model (30.8 MB)
        ├── xgb_model.pkl
        └── v2/                    # V2 tuned models (LightGBM & XGBoost)
            ├── best_model_v2.pkl  # V2 optimized model (865 KB)
            └── xgb_model_v2.pkl
```

---

## 📑 Detailed Documentation (التوثيق التفصيلي باللغة العربية)

For a highly detailed technical breakdown of the machine learning pipeline, data preprocessing (SMOTE, feature engineering), model evaluation (Youden's J, threshold tuning), SHAP values, K-Means clustering, and a comprehensive QA defense guide for the graduation committee, please read:
👉 **[Project_Documentation_AR.md](Project_Documentation_AR.md)**

---

## 🚀 Getting Started (البدء والتشغيل)

### Prerequisites
*   Python 3.8 or higher
*   Git (with LFS if downloading large assets)
*   Microsoft Power BI Desktop (to view `.pbix` dashboard)
*   SQL Server (to run `.sql` queries)

### 1. Clone the Repository
```bash
git clone https://github.com/nhahub/NHA-4-210.git
cd NHA-4-210
```

### 2. Install Python Dependencies
```bash
pip install -r "Machine Learning/requirements.txt"
```

### 3. Run the Streamlit App
The web application is located inside the `Machine Learning/` folder. Run it using:
```bash
streamlit run "Machine Learning/app.py"
```

---

## 📊 Key Highlights (أبرز نقاط المشروع)

*   **Star Schema**: Standardized data warehousing using fact tables (`fact_Diagnosis`) and dimensions (`dim_Patients`, `dim_Lifestyle`, `dim_HealthStatus`, `dim_Conditions`, `dim_Geography`, `dim_Date`).
*   **Optuna Tuning**: Model optimization using Bayesian search, improving the baseline model to achieve high clinical recall (~80.6%) on patient risks.
*   **Patient Segmentation**: K-Means clustering in Python to classify patients into distinct risk clusters, which were exported to the Power BI dashboard for health policy insights.
*   **Explainable AI**: Local risk factor contributions are displayed in the Web App using SHAP plots, showing exactly why a patient is flagged.

---

*Developed with ❤️ by **Abdelrahman Alnaggar** & **Ahmed Elsayed** — DEPI R1 Graduation Project, 2026.*
