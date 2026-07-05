import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import time
import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def main():
    t0 = time.time()
    
    print("======================================================================")
    print("STARTING DATA EXPORT & SEGMENTATION (V2 OPTIMIZED MODEL)")
    print("======================================================================")
    
    print("Loading V2 artifacts...")
    try:
        best_model = joblib.load('outputs/v2/best_model_v2.pkl')
        scaler = joblib.load('outputs/v2/scaler.pkl')
        feature_names = joblib.load('outputs/v2/feature_names.pkl')
        num_cols = joblib.load('outputs/v2/num_cols.pkl')
    except Exception as e:
        print(f"Error loading V2 artifacts: {e}")
        return
        
    print(f"Loaded best V2 model: {type(best_model).__name__}")
    
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

    X = df_ml.drop('HeartDisease', axis=1)
    
    # Ensure correct columns
    for col in feature_names:
        if col not in X.columns:
            X[col] = 0
    X = X[feature_names]
    
    X_scaled = X.copy()
    existing_num_cols = [c for c in num_cols if c in X_scaled.columns]
    X_scaled[existing_num_cols] = scaler.transform(X_scaled[existing_num_cols])
    
    print("Predicting probabilities for the entire dataset (320k rows)...")
    batch_size = 50000
    preds = []
    for i in range(0, X_scaled.shape[0], batch_size):
        preds.extend(best_model.predict_proba(X_scaled.iloc[i:i+batch_size])[:, 1])
    
    df['ML_Probability'] = preds
    # Using 0.4922 (optimal threshold from ml_pipeline_v2 logs)
    df['ML_Prediction'] = (df['ML_Probability'] >= 0.4922).astype(int)
    
    print("Running K-Means Clustering on patients...")
    cluster_features = ['BMI', 'Age_Numeric', 'Risk_Score', 'Comorbidity_Count', 'Lifestyle_Score', 'Health_Score']
    cluster_features = [f for f in cluster_features if f in df.columns]
    
    if len(cluster_features) > 0:
        X_cluster = StandardScaler().fit_transform(df[cluster_features])
        kmeans = KMeans(n_clusters=5, random_state=42)
        sample_for_kmeans = X_cluster[np.random.choice(X_cluster.shape[0], 20000, replace=False)]
        kmeans.fit(sample_for_kmeans)
        df['Patient_Segment'] = kmeans.predict(X_cluster)
        
        # Map clusters to descriptive names based on centroids
        centers = kmeans.cluster_centers_
        if 'Risk_Score' in cluster_features:
            risk_idx = cluster_features.index('Risk_Score')
            risk_scores = centers[:, risk_idx]
            sorted_clusters = np.argsort(risk_scores)
            labels = ["Very Low Risk", "Low Risk", "Moderate Risk", "High Risk", "Very High Risk"]
            mapping = {sorted_clusters[i]: labels[i] for i in range(5)}
            df['Patient_Segment_Label'] = df['Patient_Segment'].map(mapping)
    
    print("Saving enriched dataset to outputs/heart_disease_with_predictions.xlsx")
    df.to_excel('outputs/heart_disease_with_predictions.xlsx', index=False)
    
    print("\n======================================================================")
    print(f"EXPORT V2 COMPLETED IN {time.time() - t0:.1f}s")
    print("======================================================================")

if __name__ == '__main__':
    main()
