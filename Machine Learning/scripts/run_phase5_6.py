import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
import matplotlib
matplotlib.use('Agg')

import os
import time
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.inspection import PartialDependenceDisplay

def main():
    t0 = time.time()
    
    print("======================================================================")
    print("STARTING PHASE 5 & 6: XAI and Export")
    print("======================================================================")
    os.makedirs('outputs', exist_ok=True)
    
    # ─── Load Model & Artifacts ────────────────────────────────────────────────
    print("Loading artifacts...")
    try:
        best_model = joblib.load('outputs/best_model.pkl')
        scaler = joblib.load('outputs/scaler.pkl')
        feature_names = joblib.load('outputs/feature_names.pkl')
        num_cols = joblib.load('outputs/num_cols.pkl')
    except Exception as e:
        print(f"Error loading artifacts: {e}")
        return
        
    print(f"Loaded best model: {type(best_model).__name__}")
    
    # Extract XGB and RF from StackingClassifier for XAI
    xgb_model = best_model.named_estimators_['xgb']
    rf_model = best_model.named_estimators_['rf']
    
    # Save xgb_model for streamlit XAI if needed
    joblib.dump(xgb_model, 'outputs/xgb_model.pkl')
    print("Saved XGBoost model from stack as outputs/xgb_model.pkl")

    # ─── Load and transform Data ───────────────────────────────────────────────
    print("Loading original dataset...")
    df = pd.read_excel('heart_disease_project_full.xlsx', sheet_name='Main_Dataset')
    
    df_ml = df.copy()
    drop_cols = [
        'PatientID', 'State_Code', 'State', 'AgeCategory', 'Diabetic',
        'Survey_Year', 'Survey_Quarter', 'Survey_Month',
        'State_HD_vs_National', 'State_Risk_Label', 'State_Health_Tier',
        'State_Population_M',
    ]
    existing_drops = [c for c in drop_cols if c in df_ml.columns]
    df_ml = df_ml.drop(columns=existing_drops)

    binary_cols = ['HeartDisease', 'Smoking', 'AlcoholDrinking', 'Stroke', 'DiffWalking', 'PhysicalActivity', 'Diabetic_Binary', 'Asthma', 'KidneyDisease', 'SkinCancer']
    yes_no_map = {'Yes': 1, 'No': 0}
    for col in binary_cols:
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(yes_no_map).fillna(0).astype(int)

    if 'Sex' in df_ml.columns:
        df_ml['Sex'] = df_ml['Sex'].map({'Male': 1, 'Female': 0}).fillna(0).astype(int)

    ordinal_maps = {
        'GenHealth': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4},
        'BMI_Category': {'Underweight': 0, 'Normal': 1, 'Overweight': 2, 'Obese': 3, 'Obese III': 3, 'Obese II': 3, 'Obese I': 3},
        'BMI_Risk': {'Low': 0, 'Medium': 1, 'High': 2, 'Very High': 2},
        'Comorbidity_Level': {'None': 0, 'Low': 1, 'Moderate': 2, 'High': 3},
        'Risk_Tier': {'Low': 0, 'Medium': 1, 'High': 2, 'Critical': 3},
        'Lifestyle_Category': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3},
        'Sleep_Quality': {'Short': 0, 'Optimal': 1, 'Long': 2, 'Excessive': 2},
        'Age_Group': {'Young (18-34)': 0, 'Middle (35-54)': 1, 'Senior (55-69)': 2, 'Elderly (70+)': 3},
    }
    for col, mapping in ordinal_maps.items():
        if col in df_ml.columns:
            df_ml[col] = df_ml[col].map(mapping)
            median_val = df_ml[col].median()
            df_ml[col] = df_ml[col].fillna(median_val if not pd.isna(median_val) else 0).astype(int)

    ohe_cols = [c for c in ['Race', 'Region'] if c in df_ml.columns]
    if ohe_cols:
        df_ml = pd.get_dummies(df_ml, columns=ohe_cols, drop_first=True)

    df_ml.columns = [str(c) for c in df_ml.columns]

    interactions = {
        'Age_BMI_Interaction': ('Age_Numeric', 'BMI'),
        'Smoke_Diabetes': ('Smoking', 'Diabetic_Binary'),
        'Comorbidity_Age': ('Comorbidity_Count', 'Age_Numeric'),
        'Stroke_Kidney': ('Stroke', 'KidneyDisease'),
    }
    for new_col, (a, b) in interactions.items():
        if a in df_ml.columns and b in df_ml.columns:
            df_ml[new_col] = df_ml[a] * df_ml[b]

    y = df_ml['HeartDisease']
    X = df_ml.drop('HeartDisease', axis=1)
    
    # Ensure correct columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    
    X_scaled = X.copy()
    existing_num_cols = [c for c in num_cols if c in X_scaled.columns]
    X_scaled[existing_num_cols] = scaler.transform(X_scaled[existing_num_cols])
    
    # Use a sample for SHAP/Permutation to keep it fast
    print("Sampling 2000 rows for Explainable AI (XAI) analysis...")
    np.random.seed(42)
    sample_idx = np.random.choice(X_scaled.shape[0], 2000, replace=False)
    X_sample = X_scaled.iloc[sample_idx]
    y_sample = y.iloc[sample_idx]
    
    # ─── Phase 5: Explainable AI ────────────────────────────────────────────────
    print("\n--- Running SHAP Analysis (on XGBoost) ---")
    explainer = shap.TreeExplainer(xgb_model)
    shap_values = explainer.shap_values(X_sample)
    
    print("Generating SHAP global importance plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.title("SHAP Global Feature Importance")
    plt.tight_layout()
    plt.savefig("outputs/shap_global_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generating SHAP beeswarm plot...")
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Feature Impact (Beeswarm)")
    plt.tight_layout()
    plt.savefig("outputs/shap_beeswarm.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generating SHAP waterfall plot for a high-risk patient...")
    y_pred_proba = xgb_model.predict_proba(X_sample)[:, 1]
    high_risk_idx = np.argmax(y_pred_proba)
    
    plt.figure(figsize=(10, 8))
    exp = shap.Explanation(values=shap_values[high_risk_idx], 
                           base_values=explainer.expected_value, 
                           data=X_sample.iloc[high_risk_idx], 
                           feature_names=feature_names)
    shap.waterfall_plot(exp, show=False)
    plt.title("SHAP Waterfall (Highest Risk Patient)")
    plt.tight_layout()
    plt.savefig("outputs/shap_waterfall_patient.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Generating SHAP dependence plots...")
    if 'BMI' in feature_names:
        shap.dependence_plot("BMI", shap_values, X_sample, show=False)
        plt.savefig("outputs/shap_dependence_bmi.png", dpi=150, bbox_inches='tight')
        plt.close()
    if 'Age_Numeric' in feature_names:
        shap.dependence_plot("Age_Numeric", shap_values, X_sample, show=False)
        plt.savefig("outputs/shap_dependence_age.png", dpi=150, bbox_inches='tight')
        plt.close()
        
    print("\n--- Running Feature Importance Comparison ---")
    feat_imp_gini = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=False)
    
    # We use a small subset of the sample for permutation to avoid very long runtimes
    X_perm = X_sample.iloc[:500]
    y_perm = y_sample.iloc[:500]
    perm_imp = permutation_importance(best_model, X_perm, y_perm, n_repeats=3, random_state=42, n_jobs=-1)
    feat_imp_perm = pd.Series(perm_imp.importances_mean, index=feature_names).sort_values(ascending=False)
    
    feat_imp_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=feature_names).sort_values(ascending=False)
    
    imp_df = pd.DataFrame({
        'Gini (RF)': feat_imp_gini.head(15),
        'Permutation (Stacking)': feat_imp_perm.head(15),
        'SHAP Mean |val| (XGB)': feat_imp_shap.head(15)
    })
    imp_df.to_csv('outputs/feature_importance_comparison.csv')
    print("Saved feature_importance_comparison.csv")
    
    print("\n--- Generating Partial Dependence Plots ---")
    features_for_pdp = ['BMI', 'Age_Numeric', 'Comorbidity_Count']
    features_for_pdp = [f for f in features_for_pdp if f in feature_names]
    if features_for_pdp:
        fig, ax = plt.subplots(figsize=(14, 6))
        PartialDependenceDisplay.from_estimator(xgb_model, X_sample, features=features_for_pdp, ax=ax, grid_resolution=20)
        plt.suptitle("Partial Dependence Plots (XGBoost)")
        plt.tight_layout()
        plt.savefig("outputs/partial_dependence.png", dpi=150, bbox_inches='tight')
        plt.close()

    # ─── Phase 6: Export & Segmentation ────────────────────────────────────────
    print("\n======================================================================")
    print("STARTING DATA EXPORT & SEGMENTATION")
    
    print("Predicting probabilities for the entire dataset (320k rows)...")
    # Doing predictions in batches to save memory
    batch_size = 50000
    preds = []
    for i in range(0, X_scaled.shape[0], batch_size):
        preds.extend(best_model.predict_proba(X_scaled.iloc[i:i+batch_size])[:, 1])
    
    df['ML_Probability'] = preds
    # Using 0.4522 (optimal threshold from ml_pipeline logs)
    df['ML_Prediction'] = (df['ML_Probability'] >= 0.4522).astype(int)
    
    print("Running K-Means Clustering on patients...")
    cluster_features = ['BMI', 'Age_Numeric', 'Risk_Score', 'Comorbidity_Count', 'Lifestyle_Score', 'Health_Score']
    cluster_features = [f for f in cluster_features if f in df.columns]
    
    if len(cluster_features) > 0:
        X_cluster = StandardScaler().fit_transform(df[cluster_features])
        # Use a sample to fit K-means fast, then predict on all
        kmeans = KMeans(n_clusters=5, random_state=42)
        sample_for_kmeans = X_cluster[np.random.choice(X_cluster.shape[0], 20000, replace=False)]
        kmeans.fit(sample_for_kmeans)
        df['Patient_Segment'] = kmeans.predict(X_cluster)
        
        # Map clusters to descriptive names based on centroids
        centers = kmeans.cluster_centers_
        # Simplistic mapping based on Risk_Score index (2) if available
        if 'Risk_Score' in cluster_features:
            risk_idx = cluster_features.index('Risk_Score')
            risk_scores = centers[:, risk_idx]
            sorted_clusters = np.argsort(risk_scores)
            labels = ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
            mapping = {sorted_clusters[i]: labels[i] for i in range(5)}
            df['Patient_Segment_Label'] = df['Patient_Segment'].map(mapping)
    
    print("Saving enriched dataset to outputs/heart_disease_with_predictions.xlsx")
    # Export for Power BI
    df.to_excel('outputs/heart_disease_with_predictions.xlsx', index=False)
    
    print("\n======================================================================")
    print(f"ALL TASKS COMPLETED IN {time.time() - t0:.1f}s")
    print("======================================================================")

if __name__ == '__main__':
    main()
