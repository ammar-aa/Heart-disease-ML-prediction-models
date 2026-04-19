import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import shap

model, feature_names = joblib.load("Heart_Disease_Project/ui/heart_model.pkl")

st.set_page_config(page_title="Heart Check-up", layout="centered")
st.title("AI Heart Disease Prediction")

with st.sidebar:
    st.header("📖 Inputs Guide")
    st.markdown("""
    - **Sex:** 1=Male, 0=Female
    - **Chest Pain (cp):** 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic
    - **Thalassemia (thal):** 1=Normal, 2=Fixed, 3=Reversible
    - **Slope:** 0=Upsloping, 1=Flat, 2=Downsloping
    - **Exang:** 1=Yes, 0=No
    """)

st.write("### 🩺 Patient Vital Data")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex (1:M, 0:F)", [1, 0])
    cp = st.selectbox("Chest Pain (0-3)", [0, 1, 2, 3])
    thalach = st.number_input("Max Heart Rate", 40, 220, 150)

with col2:
    oldpeak = st.number_input("ST Depression", 0.0, 7.0, 0.0)
    ca = st.slider("Major Vessels", 0, 3, 0)
    thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])
    slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])
    exang = st.selectbox("Exercise Angina (1:Y, 0:N)", [0, 1])

if st.button("Predict"):
    input_data = {
        "thal": thal, "ca": ca, "slope": slope, "thalach": thalach,
        "exang": exang, "sex": sex, "oldpeak": oldpeak, "cp": cp, "age": age
    }
    input_df = pd.DataFrame([input_data])[feature_names]

    proba = model.predict_proba(input_df)
    percent_sick = proba[0][1] * 100  
    
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent_sick,
        number={'suffix': "%", 'font': {'size': 60}},
        title={'text': "Heart Disease Risk Level (%)"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "white"}, 
            'steps': [
                {'range': [0, 30], 'color': "#00ff00"},    
                {'range': [30, 70], 'color': "#ffff00"},  
                {'range': [70, 100], 'color': "#ff0000"}  
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': percent_sick
            }
        }
    ))
    st.plotly_chart(fig_gauge)

    if percent_sick > 50:
        st.error(f"High Risk Detected: {percent_sick:.2f}%")
    else:
        st.success(f"Low Risk Score: {percent_safe:.2f}%")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent_sick,
        number={'suffix': "%", 'font': {'size': 60}},   
        title={'text': "Heart Disease Risk Level", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [0, 100], 'ticksuffix': "%"}, 
            'bar': {'color': "white"}, 
            'steps': [
                {'range': [0, 30], 'color': "#00ff00"},
                {'range': [30, 70], 'color': "#ffff00"},
                {'range': [70, 100], 'color': "#ff0000"}
            ]
        }
    ))
    st.plotly_chart(fig_gauge)

    st.write("---")
    st.subheader("🔍 Feature Importance Analysis")

    try:
        with st.spinner('Deep Analysis...'):
            f = lambda x: model.predict_proba(pd.DataFrame(x, columns=feature_names))
            explainer = shap.KernelExplainer(f, input_df.values)
            raw_sv = explainer.shap_values(input_df.values, nsamples=200)

            if isinstance(raw_sv, list):
                sv = np.array(raw_sv[1]).flatten() if len(raw_sv) > 1 else np.array(raw_sv[0]).flatten()
            else:
                sv = np.array(raw_sv).flatten()

            impact_df = pd.DataFrame({'Feature': feature_names, 'Impact': sv})
            
            max_abs_impact = np.abs(impact_df['Impact']).max()
            if max_abs_impact < 0.1 and max_abs_impact > 0:
                scale_factor = 0.5 / max_abs_impact
                impact_df['Impact'] = impact_df['Impact'] * scale_factor
                st.caption(f"Note: Visual impact bars scaled for clarity.")

            impact_df = impact_df.sort_values(by='Impact')

            fig_impact = px.bar(
                impact_df, x='Impact', y='Feature', orientation='h',
                color='Impact',
                color_continuous_scale='RdBu_r',
                template='plotly_dark'
            )
            
            limit = max(abs(impact_df['Impact']).max(), 0.001) * 1.2
            fig_impact.update_layout(xaxis=dict(range=[-limit, limit], zeroline=True))
            
            st.plotly_chart(fig_impact, use_container_width=True)

    except Exception as e:
        st.warning(f"Note: Analysis visualization is recalculating. Prediction: {percent_sick:.2f}%")
