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
    data_dict = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "thalach": thalach,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal
    }
    
    input_df = pd.DataFrame([data_dict])[feature_names]

    proba = model.predict_proba(input_df)
    
    percent_sick = proba[0][1] * 100
    percent_safe = proba[0][0] * 100

    if percent_sick > 50:
        st.error(f"⚠️ High Risk Detected: {percent_sick:.2f}%")
    else:
        st.success(f"✅ Low Risk Score: {percent_safe:.2f}% safe")

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=percent_sick,
        number={'suffix': "%"},
        title={'text': "Danger Level"},
        gauge={
            'axis': {'range': [0, 100]},
            'steps': [
                {'range': [0, 30], 'color': "green"},
                {'range': [30, 70], 'color': "yellow"},
                {'range': [70, 100], 'color': "red"}
            ],
            'bar': {'color': "black"}
        }
    ))
    st.plotly_chart(fig_gauge)
