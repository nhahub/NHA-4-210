"""
Heart Disease Risk Prediction — Streamlit Web App
===================================================
Loads the best trained model and provides an interactive UI
for predicting heart disease risk for new patients, with SHAP explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─── Page configuration ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main { font-family: 'Inter', sans-serif; }
    
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #E74C3C, #C0392B, #E74C3C);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        text-align: center;
        color: #7f8c8d;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    
    .risk-card {
        border-radius: 16px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.1);
    }
    .risk-low { background: linear-gradient(135deg, #27ae6020, #2ecc7130); border-color: #27ae60; }
    .risk-medium { background: linear-gradient(135deg, #f39c1220, #e67e2230); border-color: #f39c12; }
    .risk-high { background: linear-gradient(135deg, #e74c3c20, #c0392b30); border-color: #e74c3c; }
    .risk-critical { background: linear-gradient(135deg, #8e44ad20, #6c348330); border-color: #8e44ad; }
    
    .risk-value {
        font-size: 3.5rem;
        font-weight: 700;
        margin: 0.5rem 0;
    }
    .risk-label {
        font-size: 1.3rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 2px;
    }
    
    .metric-card {
        background: rgba(255,255,255,0.05);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #E74C3C, #C0392B);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(231, 76, 60, 0.3);
    }
    
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        margin: 1.5rem 0 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid rgba(231, 76, 60, 0.3);
    }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #16213e 100%);
    }
</style>
""", unsafe_allow_html=True)


# ─── Load model and artifacts ──────────────────────────────────────────────
@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    
    # Load V1
    try:
        v1_model = joblib.load('outputs/best_model.pkl')
        v1_xgb = joblib.load('outputs/xgb_model.pkl')
        v1_scaler = joblib.load('outputs/scaler.pkl')
        v1_features = joblib.load('outputs/feature_names.pkl')
        v1_num_cols = joblib.load('outputs/num_cols.pkl')
        v1_comp = None
        if os.path.exists('outputs/model_comparison.csv'):
            v1_comp = pd.read_csv('outputs/model_comparison.csv')
        artifacts['V1'] = {
            'model': v1_model, 'xgb': v1_xgb, 'scaler': v1_scaler,
            'features': v1_features, 'num_cols': v1_num_cols, 'comparison': v1_comp
        }
    except Exception as e:
        st.warning(f"⚠️ Could not load V1 (Baseline) artifacts: {e}")
        
    # Load V2
    try:
        v2_model = joblib.load('outputs/v2/best_model_v2.pkl')
        v2_xgb = joblib.load('outputs/v2/xgb_model_v2.pkl')
        v2_scaler = joblib.load('outputs/v2/scaler.pkl')
        v2_features = joblib.load('outputs/v2/feature_names.pkl')
        v2_num_cols = joblib.load('outputs/v2/num_cols.pkl')
        v2_comp = None
        if os.path.exists('outputs/v2/model_comparison_v2.csv'):
            v2_comp = pd.read_csv('outputs/v2/model_comparison_v2.csv')
        artifacts['V2'] = {
            'model': v2_model, 'xgb': v2_xgb, 'scaler': v2_scaler,
            'features': v2_features, 'num_cols': v2_num_cols, 'comparison': v2_comp
        }
    except Exception as e:
        st.warning(f"⚠️ Could not load V2 (Optimized) artifacts: {e}")
        
    return artifacts


artifacts = load_all_artifacts()
if not artifacts:
    st.error("❌ No model artifacts could be loaded. Please run the ML pipelines first.")
    st.stop()

# ─── Hero Section ───────────────────────────────────────────────────────────
st.markdown('<div class="hero-title">🫀 Heart Disease Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">AI-Powered Prediction with Explainable Results • Graduation Project 2026</div>', unsafe_allow_html=True)

# ─── Sidebar: Patient Input ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Model Settings")
    available_versions = list(artifacts.keys())
    # Prefer V2 if available, otherwise V1
    default_idx = available_versions.index('V2') if 'V2' in available_versions else 0
    version_label = st.selectbox("Select Model Version", 
                                 [f"{v} - {'Optimized (LightGBM)' if v=='V2' else 'Baseline (Stacking)'}" for v in available_versions], 
                                 index=default_idx)
    selected_ver = version_label.split(' - ')[0]
    
    st.markdown("---")
    st.markdown("## 👤 Patient Information")
    
    # Demographics
    st.markdown("### Demographics")
    age = st.slider("Age", 18, 90, 50, key="age")
    sex = st.selectbox("Sex", ["Male", "Female"], key="sex")
    race = st.selectbox("Race", ["White", "Black", "Hispanic", "Asian", "American Indian/Alaskan Native", "Other"], key="race")
    
    st.markdown("### Health Metrics")
    bmi = st.slider("BMI", 12.0, 95.0, 28.0, 0.1, key="bmi")
    gen_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"], key="gen_health")
    physical_health = st.slider("Poor Physical Health Days (last 30)", 0, 30, 3, key="physical_health")
    mental_health = st.slider("Poor Mental Health Days (last 30)", 0, 30, 3, key="mental_health")
    sleep_time = st.slider("Sleep Hours/Night", 1, 24, 7, key="sleep_time")
    diff_walking = st.selectbox("Difficulty Walking", ["No", "Yes"], key="diff_walking")
    
    st.markdown("### Medical History")
    smoking = st.selectbox("Smoking (100+ cigarettes ever)", ["No", "Yes"], key="smoking")
    alcohol = st.selectbox("Heavy Alcohol Drinking", ["No", "Yes"], key="alcohol")
    stroke = st.selectbox("History of Stroke", ["No", "Yes"], key="stroke")
    diabetes = st.selectbox("Diabetes", ["No", "Yes"], key="diabetes")
    physical_activity = st.selectbox("Physical Activity (last 30 days)", ["Yes", "No"], key="physical_activity")
    asthma = st.selectbox("Asthma", ["No", "Yes"], key="asthma")
    kidney_disease = st.selectbox("Kidney Disease", ["No", "Yes"], key="kidney_disease")
    skin_cancer = st.selectbox("Skin Cancer", ["No", "Yes"], key="skin_cancer")

    predict_button = st.button("🔍 Predict Risk", type="primary", key="predict")

# ─── Retrieve active model version ──────────────────────────────────────────
model = artifacts[selected_ver]['model']
xgb_model = artifacts[selected_ver]['xgb']
scaler = artifacts[selected_ver]['scaler']
feature_names = artifacts[selected_ver]['features']
num_cols = artifacts[selected_ver]['num_cols']
comparison_df = artifacts[selected_ver]['comparison']

# ─── Prediction Logic ──────────────────────────────────────────────────────
def prepare_patient_data():
    """Build a DataFrame matching the trained feature set."""
    # Binary mappings
    yes_no = lambda x: 1 if x == "Yes" else 0
    sex_val = 1 if sex == "Male" else 0
    
    # Ordinal mappings
    gen_health_map = {'Poor': 0, 'Fair': 1, 'Good': 2, 'Very good': 3, 'Excellent': 4}
    
    # BMI category
    if bmi < 18.5:
        bmi_cat, bmi_risk = 0, 0
    elif bmi < 25:
        bmi_cat, bmi_risk = 1, 0
    elif bmi < 30:
        bmi_cat, bmi_risk = 2, 1
    else:
        bmi_cat, bmi_risk = 3, 2
    
    # Age group
    if age < 35:
        age_group = 0
    elif age < 55:
        age_group = 1
    elif age < 70:
        age_group = 2
    else:
        age_group = 3
    
    # Comorbidities
    comorbidities = [yes_no(stroke), yes_no(diabetes), yes_no(asthma), 
                     yes_no(kidney_disease), yes_no(skin_cancer)]
    comorbidity_count = sum(comorbidities)
    
    if comorbidity_count == 0:
        comorbidity_level = 0
    elif comorbidity_count == 1:
        comorbidity_level = 1
    elif comorbidity_count == 2:
        comorbidity_level = 2
    else:
        comorbidity_level = 3
    
    # Risk score (simplified)
    risk_score = (comorbidity_count * 1.0 + 
                  (1 if bmi > 30 else 0) * 0.8 +
                  yes_no(smoking) * 0.7 +
                  yes_no(stroke) * 1.5 +
                  yes_no(diabetes) * 1.0 +
                  (1 if age > 60 else 0) * 0.5 +
                  (1 if gen_health_map[gen_health] < 2 else 0) * 0.8)
    
    risk_tier = 0 if risk_score < 2 else (1 if risk_score < 4 else (2 if risk_score < 6 else 3))
    
    # Lifestyle
    lifestyle_score = 100
    if yes_no(smoking): lifestyle_score -= 25
    if yes_no(alcohol): lifestyle_score -= 10
    if not yes_no(physical_activity): lifestyle_score -= 25
    if bmi > 30: lifestyle_score -= 15
    lifestyle_score = max(0, lifestyle_score)
    lifestyle_cat = 0 if lifestyle_score < 25 else (1 if lifestyle_score < 50 else (2 if lifestyle_score < 75 else 3))
    
    # Health score
    health_score = (gen_health_map[gen_health] * 4 + 
                    (30 - physical_health) / 3 + 
                    (30 - mental_health) / 3)
    
    # Sleep quality
    sleep_qual = 0 if sleep_time < 6 else (1 if sleep_time <= 9 else 2)
    
    # Build feature dict
    data = {
        'BMI': bmi,
        'Smoking': yes_no(smoking),
        'AlcoholDrinking': yes_no(alcohol),
        'Stroke': yes_no(stroke),
        'PhysicalHealth': physical_health,
        'MentalHealth': mental_health,
        'DiffWalking': yes_no(diff_walking),
        'Sex': sex_val,
        'PhysicalActivity': yes_no(physical_activity),
        'GenHealth': gen_health_map[gen_health],
        'SleepTime': sleep_time,
        'Asthma': yes_no(asthma),
        'KidneyDisease': yes_no(kidney_disease),
        'SkinCancer': yes_no(skin_cancer),
        'Age_Numeric': age,
        'Age_Group': age_group,
        'BMI_Category': bmi_cat,
        'BMI_Risk': bmi_risk,
        'Diabetic_Binary': yes_no(diabetes),
        'Comorbidity_Count': comorbidity_count,
        'Comorbidity_Level': comorbidity_level,
        'Risk_Score': risk_score,
        'Risk_Tier': risk_tier,
        'Lifestyle_Score': lifestyle_score,
        'Lifestyle_Category': lifestyle_cat,
        'Health_Score': health_score,
        'Sleep_Quality': sleep_qual,
        'State_HD_Prevalence': 8.5,  # National average
        'State_Obesity_Rate': 31.9,
        'State_Smoking_Rate': 14.0,
        'State_Uninsured_Rate': 9.2,
        'State_Median_Income_K': 65,
        'State_Health_Rank': 25,
    }
    
    # Race one-hot
    race_cols = ['Race_Black', 'Race_Hispanic', 'Race_Other', 'Race_White']
    for rc in race_cols:
        data[rc] = 0
    race_map = {
        'Black': 'Race_Black', 'Hispanic': 'Race_Hispanic',
        'Other': 'Race_Other', 'White': 'Race_White',
        'Asian': 'Race_Other', 'American Indian/Alaskan Native': 'Race_Other'
    }
    if race in race_map and race_map[race] in data:
        data[race_map[race]] = 1
    
    # Region one-hot (default: national avg)
    region_cols = ['Region_Northeast', 'Region_South', 'Region_West']
    for rc in region_cols:
        data[rc] = 0
    
    # Interaction features
    data['Age_BMI_Interaction'] = age * bmi
    data['Smoke_Diabetes'] = yes_no(smoking) * yes_no(diabetes)
    data['Comorbidity_Age'] = comorbidity_count * age
    data['Stroke_Kidney'] = yes_no(stroke) * yes_no(kidney_disease)
    
    # Create DataFrame with correct column order
    patient_df = pd.DataFrame([data])
    
    # Ensure all feature columns exist
    for col in feature_names:
        if col not in patient_df.columns:
            patient_df[col] = 0
    
    patient_df = patient_df[feature_names]
    
    # Scale numerical columns
    scale_cols = [c for c in num_cols if c in patient_df.columns]
    if scale_cols:
        patient_df[scale_cols] = scaler.transform(patient_df[scale_cols])
    
    return patient_df


# ─── Main Content ───────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔍 Prediction", "📊 Model Performance", "ℹ️ About"])

with tab1:
    if predict_button:
        with st.spinner("Analyzing patient data..."):
            patient_df = prepare_patient_data()
            
            # Get prediction
            probability = model.predict_proba(patient_df)[0][1]
            risk_pct = probability * 100
            
            # Determine risk level
            if risk_pct < 10:
                risk_level, risk_class, risk_color, risk_emoji = "LOW", "risk-low", "#27ae60", "✅"
            elif risk_pct < 30:
                risk_level, risk_class, risk_color, risk_emoji = "MEDIUM", "risk-medium", "#f39c12", "⚠️"
            elif risk_pct < 50:
                risk_level, risk_class, risk_color, risk_emoji = "HIGH", "risk-high", "#e74c3c", "🔴"
            else:
                risk_level, risk_class, risk_color, risk_emoji = "CRITICAL", "risk-critical", "#8e44ad", "🚨"
        
        # Determine explicit diagnosis based on optimal threshold (V1: 45%, V2: 49.22%)
        threshold = 49.22 if selected_ver == 'V2' else 45.0
        if risk_pct >= threshold:
            diagnosis_text = "⚠️ POSITIVE (Heart Disease Detected)"
            diagnosis_color = "#e74c3c"
        else:
            diagnosis_text = "✅ NEGATIVE (Healthy / No Heart Disease)"
            diagnosis_color = "#27ae60"

        # Display result
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown(f"""
            <div class="risk-card {risk_class}">
                <div style="font-size: 1.5rem; font-weight: 800; color: {diagnosis_color}; margin-bottom: 10px;">
                    {diagnosis_text}
                </div>
                <hr style="border-color: rgba(255,255,255,0.1); margin: 10px 0;">
                <div class="risk-label">{risk_emoji} Risk Level: {risk_level}</div>
                <div class="risk-value" style="color: {risk_color};">{risk_pct:.1f}%</div>
                <p style="color: #888; font-size: 0.9rem;">Probability of Heart Disease</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Patient summary
        st.markdown('<div class="section-header">📋 Patient Summary</div>', unsafe_allow_html=True)
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Age", f"{age} years")
            st.metric("BMI", f"{bmi:.1f}")
        with c2:
            st.metric("General Health", gen_health)
            st.metric("Sleep", f"{sleep_time}h/night")
        with c3:
            st.metric("Smoking", smoking)
            st.metric("Diabetes", diabetes)
        with c4:
            st.metric("Stroke History", stroke)
            st.metric("Kidney Disease", kidney_disease)
        
        # Key risk factors
        st.markdown('<div class="section-header">🎯 Key Risk Factors</div>', unsafe_allow_html=True)
        
        risk_factors = []
        if bmi > 30: risk_factors.append(("🍔 Obesity", f"BMI = {bmi:.1f} (Obese)", "High"))
        if smoking == "Yes": risk_factors.append(("🚬 Smoking", "Active smoker", "High"))
        if stroke == "Yes": risk_factors.append(("🧠 Stroke History", "Previous stroke", "Critical"))
        if diabetes == "Yes": risk_factors.append(("💉 Diabetes", "Diabetic", "High"))
        if kidney_disease == "Yes": risk_factors.append(("🫘 Kidney Disease", "Has kidney disease", "High"))
        if age > 65: risk_factors.append(("👴 Advanced Age", f"Age = {age}", "Medium"))
        if gen_health in ["Poor", "Fair"]: risk_factors.append(("📉 Poor Health", f"Self-rated: {gen_health}", "Medium"))
        if physical_health > 15: risk_factors.append(("🤕 Poor Physical Health", f"{physical_health} bad days/month", "Medium"))
        if physical_activity == "No": risk_factors.append(("🏋️ Sedentary", "No physical activity", "Medium"))
        
        if risk_factors:
            for icon_name, description, severity in risk_factors:
                color = "#e74c3c" if severity == "Critical" else ("#f39c12" if severity == "High" else "#3498db")
                st.markdown(f"- **{icon_name}**: {description} `{severity}`")
        else:
            st.success("✅ No major risk factors identified!")
        
        # SHAP Explanation
        st.markdown('<div class="section-header">🧠 AI Reasoning (SHAP)</div>', unsafe_allow_html=True)
        st.markdown("This chart shows exactly which factors increased (🔴 red) or decreased (🔵 blue) the AI's risk prediction for this specific patient.")
        
        try:
            import shap
            
            # Use TreeExplainer on XGBoost model
            explainer = shap.TreeExplainer(xgb_model)
            shap_values = explainer.shap_values(patient_df)
            
            # Get top features by absolute SHAP value
            sv = shap_values[0]
            feat_importance = pd.DataFrame({
                'feature': feature_names,
                'shap_value': sv,
                'abs_shap': np.abs(sv)
            }).sort_values('abs_shap', ascending=False).head(15)
            
            # Sort for waterfall display (ascending for horizontal bar)
            feat_importance = feat_importance.sort_values('shap_value', ascending=True)
            
            colors = ['#E74C3C' if v > 0 else '#3498DB' for v in feat_importance['shap_value']]
            
            fig_shap = go.Figure(go.Bar(
                x=feat_importance['shap_value'],
                y=feat_importance['feature'],
                orientation='h',
                marker_color=colors,
                text=[f"{v:+.3f}" for v in feat_importance['shap_value']],
                textposition='outside',
                textfont=dict(size=11),
            ))
            
            fig_shap.update_layout(
                title=dict(
                    text=f"SHAP Feature Impact (Base value: {explainer.expected_value:.3f})",
                    font=dict(size=16)
                ),
                xaxis_title="SHAP Value (impact on prediction)",
                yaxis_title="",
                template="plotly_dark",
                height=500,
                margin=dict(l=180, r=80, t=60, b=50),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(zeroline=True, zerolinecolor='rgba(255,255,255,0.3)', zerolinewidth=2,
                           gridcolor='rgba(255,255,255,0.05)'),
                yaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            )
            
            st.plotly_chart(fig_shap, use_container_width=True)
            
            # Also show a summary with the base value explanation
            st.caption(f"📐 Base value = {explainer.expected_value:.3f} | "
                       f"Final prediction = {explainer.expected_value + sv.sum():.3f} | "
                       f"Total SHAP shift = {sv.sum():+.3f}")
            
        except Exception as e:
            st.warning(f"Could not generate SHAP explanation: {e}")
        
        # Recommendations
        st.markdown('<div class="section-header">💡 Recommendations</div>', unsafe_allow_html=True)
        
        if risk_pct >= 30:
            st.error("⚠️ **High Risk Detected** — Please consult a cardiologist for a comprehensive heart health evaluation.")
        if smoking == "Yes":
            st.warning("🚭 **Quit Smoking** — This is the single most impactful change for heart health.")
        if bmi > 30:
            st.warning("🏃 **Weight Management** — A 5-10% weight reduction can significantly lower heart disease risk.")
        if physical_activity == "No":
            st.info("🏋️ **Start Exercising** — Aim for at least 150 minutes of moderate activity per week.")
        if diabetes == "Yes":
            st.info("💊 **Blood Sugar Control** — Maintain HbA1c below 7% with your doctor's guidance.")
        if risk_pct < 10:
            st.success("🎉 **Keep it up!** — Continue your healthy lifestyle to maintain low risk.")

    else:
        # Landing state
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("### 🎯 Accurate")
            st.markdown("Trained on **319,795** patients with 6 ML models including Deep Learning")
        with col2:
            st.markdown("### 🔬 Explainable")
            st.markdown("SHAP-powered explanations show **why** each prediction was made")
        with col3:
            st.markdown("### ⚡ Instant")
            st.markdown("Get results in **milliseconds** with actionable health recommendations")
        
        st.markdown("---")
        st.info("👈 **Fill in patient data in the sidebar and click 'Predict Risk'** to get started!")

with tab2:
    st.markdown(f'<div class="section-header">📊 Model Performance Comparison ({selected_ver})</div>', unsafe_allow_html=True)
    
    if comparison_df is not None:
        st.markdown(f"### {selected_ver} Model Leaderboard")
        st.dataframe(comparison_df, use_container_width=True)
        
        # ── Interactive Plotly Charts from comparison_df ──
        # Detect model name column and metric columns
        comp_cols = comparison_df.columns.tolist()
        model_col = comp_cols[0]  # First column is typically the model name
        metric_cols = [c for c in comp_cols if c != model_col and comparison_df[c].dtype in ['float64', 'float32', 'int64']]
        
        if metric_cols:
            # 1) Grouped Bar Chart — All metrics per model
            st.markdown("### 📊 Model Metrics Comparison")
            
            bar_colors = px.colors.qualitative.Set2
            fig_bar = go.Figure()
            for i, metric in enumerate(metric_cols):
                fig_bar.add_trace(go.Bar(
                    name=metric,
                    x=comparison_df[model_col],
                    y=comparison_df[metric],
                    text=[f"{v:.4f}" if v < 1.5 else f"{v:.1f}" for v in comparison_df[metric]],
                    textposition='outside',
                    marker_color=bar_colors[i % len(bar_colors)],
                ))
            
            fig_bar.update_layout(
                barmode='group',
                template='plotly_dark',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(title='Model', tickangle=-25),
                yaxis=dict(title='Score', gridcolor='rgba(255,255,255,0.05)'),
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                margin=dict(t=80, b=80),
            )
            st.plotly_chart(fig_bar, use_container_width=True)
            
            # 2) Radar Chart — Top models comparison
            # Filter to metrics that are between 0 and 1 (percentages/scores)
            radar_metrics = [c for c in metric_cols if comparison_df[c].max() <= 1.05]
            if len(radar_metrics) >= 3:
                st.markdown("### 🕸️ Model Radar Comparison")
                fig_radar = go.Figure()
                radar_colors = px.colors.qualitative.Plotly
                for i, (_, row) in enumerate(comparison_df.iterrows()):
                    values = [row[m] for m in radar_metrics]
                    values.append(values[0])  # close the polygon
                    fig_radar.add_trace(go.Scatterpolar(
                        r=values,
                        theta=radar_metrics + [radar_metrics[0]],
                        fill='toself',
                        name=str(row[model_col]),
                        line=dict(color=radar_colors[i % len(radar_colors)]),
                        opacity=0.7,
                    ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1], gridcolor='rgba(255,255,255,0.1)'),
                        bgcolor='rgba(0,0,0,0)',
                    ),
                    template='plotly_dark',
                    height=550,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', yanchor='bottom', y=-0.2, xanchor='center', x=0.5),
                    margin=dict(t=40, b=80),
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            
            # 3) Highlight Best Model with a Plotly Indicator
            # Find best model by first metric column (usually Accuracy or ROC-AUC)
            best_metric = metric_cols[0]
            best_idx = comparison_df[best_metric].idxmax()
            best_model_name = comparison_df.loc[best_idx, model_col]
            best_score = comparison_df.loc[best_idx, best_metric]
            
            st.markdown(f"### 🏆 Best Model: **{best_model_name}**")
            
            indicator_cols = st.columns(min(len(metric_cols), 4))
            for i, metric in enumerate(metric_cols[:4]):
                with indicator_cols[i]:
                    val = comparison_df.loc[best_idx, metric]
                    fig_ind = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=val * 100 if val <= 1 else val,
                        title={'text': metric, 'font': {'size': 14}},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': '#E74C3C'},
                            'bgcolor': 'rgba(0,0,0,0)',
                            'steps': [
                                {'range': [0, 50], 'color': 'rgba(231,76,60,0.1)'},
                                {'range': [50, 75], 'color': 'rgba(243,156,18,0.1)'},
                                {'range': [75, 100], 'color': 'rgba(39,174,96,0.1)'},
                            ],
                        },
                        number={'suffix': '%' if val <= 1 else '', 'font': {'size': 20}},
                    ))
                    fig_ind.update_layout(
                        height=200,
                        margin=dict(t=40, b=10, l=20, r=20),
                        template='plotly_dark',
                        paper_bgcolor='rgba(0,0,0,0)',
                    )
                    st.plotly_chart(fig_ind, use_container_width=True)
        
        # ── ROC Curves (Plotly) ──
        st.markdown("### 📈 ROC Curves")
        if selected_ver == 'V2':
            roc_csv = 'outputs/v2/roc_curves_data_v2.csv'
        else:
            roc_csv = 'outputs/roc_curves_data.csv'
        
        if os.path.exists(roc_csv):
            roc_df = pd.read_csv(roc_csv)
            roc_colors = px.colors.qualitative.Set1
            fig_roc = go.Figure()
            for i, model_name in enumerate(roc_df['Model'].unique()):
                model_data = roc_df[roc_df['Model'] == model_name]
                auc_val = model_data['AUC'].iloc[0]
                fig_roc.add_trace(go.Scatter(
                    x=model_data['FPR'], y=model_data['TPR'],
                    mode='lines',
                    name=f"{model_name} (AUC={auc_val:.4f})",
                    line=dict(color=roc_colors[i % len(roc_colors)], width=2),
                ))
            # Diagonal reference
            fig_roc.add_trace(go.Scatter(
                x=[0, 1], y=[0, 1], mode='lines',
                line=dict(color='rgba(255,255,255,0.3)', dash='dash', width=1),
                showlegend=False,
            ))
            fig_roc.update_layout(
                title=f'Combined ROC Curves — {selected_ver} Models',
                xaxis=dict(title='False Positive Rate', gridcolor='rgba(255,255,255,0.05)', range=[0, 1]),
                yaxis=dict(title='True Positive Rate', gridcolor='rgba(255,255,255,0.05)', range=[0, 1.02]),
                template='plotly_dark',
                height=500,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(x=0.55, y=0.05, bgcolor='rgba(0,0,0,0.5)'),
            )
            st.plotly_chart(fig_roc, use_container_width=True)
        else:
            # Fallback to static image
            img_path = 'outputs/v2/combined_roc_curves_v2.png' if selected_ver == 'V2' else 'outputs/combined_roc_curves.png'
            if os.path.exists(img_path):
                st.image(img_path, caption=f"Combined ROC Curves — {selected_ver} Models", use_container_width=True)
        
        # ── V1 vs V2 Comparison (Plotly) ──
        if selected_ver == 'V2':
            st.markdown("### 🔄 V1 vs V2 Best Model Comparison")
            v1_comp_path = 'outputs/model_comparison.csv'
            if os.path.exists(v1_comp_path):
                v1_df = pd.read_csv(v1_comp_path)
                v1_best = v1_df.loc[v1_df['ROC-AUC'].idxmax()]
                v2_best_row = comparison_df.loc[comparison_df['ROC-AUC'].idxmax()] if 'ROC-AUC' in comparison_df.columns else comparison_df.iloc[0]
                
                metrics_list = ['ROC-AUC', 'PR-AUC', 'F1-Score', 'Recall', 'Precision', 'Accuracy']
                metrics_list = [m for m in metrics_list if m in v1_best.index and m in v2_best_row.index]
                
                v1_vals = [v1_best[m] for m in metrics_list]
                v2_vals = [v2_best_row[m] for m in metrics_list]
                
                fig_comp = go.Figure()
                fig_comp.add_trace(go.Bar(
                    name=f"V1: {v1_best['Model']}",
                    x=metrics_list, y=v1_vals,
                    text=[f"{v:.3f}" for v in v1_vals],
                    textposition='outside',
                    marker_color='#90CAF9',
                    marker_line=dict(color='#1565C0', width=1.5),
                ))
                fig_comp.add_trace(go.Bar(
                    name=f"V2: {v2_best_row['Model']}",
                    x=metrics_list, y=v2_vals,
                    text=[f"{v:.3f}" for v in v2_vals],
                    textposition='outside',
                    marker_color='#EF9A9A',
                    marker_line=dict(color='#C62828', width=1.5),
                ))
                fig_comp.update_layout(
                    barmode='group',
                    title='V1 Baseline vs V2 Optimized — Best Model Comparison',
                    yaxis=dict(title='Score', range=[0, 1.15], gridcolor='rgba(255,255,255,0.05)'),
                    template='plotly_dark',
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
                )
                st.plotly_chart(fig_comp, use_container_width=True)
        
        # ── Learning Curves (Plotly) ──
        if selected_ver == 'V2':
            lc_csv = 'outputs/v2/learning_curve_data_v2.csv'
            if os.path.exists(lc_csv):
                st.markdown("### 📈 Learning Curves (Overfitting & Underfitting Diagnosis)")
                lc_data = pd.read_csv(lc_csv)
                
                fig_lc = go.Figure()
                # Training score band
                fig_lc.add_trace(go.Scatter(
                    x=pd.concat([lc_data['train_size'], lc_data['train_size'][::-1]]),
                    y=pd.concat([lc_data['train_mean'] + lc_data['train_std'],
                                 (lc_data['train_mean'] - lc_data['train_std'])[::-1]]),
                    fill='toself', fillcolor='rgba(33,150,243,0.1)',
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False,
                ))
                fig_lc.add_trace(go.Scatter(
                    x=lc_data['train_size'], y=lc_data['train_mean'],
                    mode='lines+markers', name='Training Score',
                    line=dict(color='#2196F3', width=2), marker=dict(size=8),
                ))
                # CV score band
                fig_lc.add_trace(go.Scatter(
                    x=pd.concat([lc_data['train_size'], lc_data['train_size'][::-1]]),
                    y=pd.concat([lc_data['test_mean'] + lc_data['test_std'],
                                 (lc_data['test_mean'] - lc_data['test_std'])[::-1]]),
                    fill='toself', fillcolor='rgba(244,67,54,0.1)',
                    line=dict(color='rgba(0,0,0,0)'), showlegend=False,
                ))
                fig_lc.add_trace(go.Scatter(
                    x=lc_data['train_size'], y=lc_data['test_mean'],
                    mode='lines+markers', name='Cross-Validation Score',
                    line=dict(color='#F44336', width=2), marker=dict(size=8),
                ))
                fig_lc.update_layout(
                    title='Learning Curve — XGBoost V2 (Tuned)',
                    xaxis=dict(title='Training Set Size', gridcolor='rgba(255,255,255,0.05)'),
                    yaxis=dict(title='ROC-AUC Score', gridcolor='rgba(255,255,255,0.05)'),
                    template='plotly_dark',
                    height=450,
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    legend=dict(x=0.65, y=0.05, bgcolor='rgba(0,0,0,0.5)'),
                )
                st.plotly_chart(fig_lc, use_container_width=True)
            else:
                lc_img = 'outputs/v2/learning_curve_xgb_v2.png'
                if os.path.exists(lc_img):
                    st.markdown("### 📈 Learning Curves (Overfitting & Underfitting Diagnosis)")
                    st.image(lc_img, caption="Learning Curves — XGBoost V2 (Tuned)", use_container_width=True)
    else:
        st.warning(f"Model comparison data for {selected_ver} not available.")

with tab3:
    st.markdown("""
    ### About This Project
    
    This is a **graduation project** for predicting heart disease risk using Machine Learning.
    
    **Dataset:** CDC Behavioral Risk Factor Surveillance System (BRFSS)  
    **Patients:** 319,795  
    **Features:** 48 (including engineered features)  
    **Models Used:**
    1. Logistic Regression (Baseline)
    2. Random Forest
    3. XGBoost
    4. LightGBM
    5. Neural Network (MLP)
    6. Stacking Ensemble
    
    **Key Techniques:**
    - SMOTE for handling class imbalance (9% positive rate)
    - Stratified K-Fold Cross-Validation
    - Threshold tuning via Youden's J statistic
    - SHAP & LIME for model interpretability
    
    ---
    *Built with ❤️ using Python, scikit-learn, XGBoost, and Streamlit*
    """)

# ─── Footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #888; font-size: 0.8rem;'>"
    "Heart Disease Risk Predictor • Graduation Project 2026 • "
    "Powered by Machine Learning & Explainable AI"
    "</p>",
    unsafe_allow_html=True
)
