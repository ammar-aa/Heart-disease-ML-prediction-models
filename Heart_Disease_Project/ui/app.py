import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import numpy as np
import plotly.graph_objects as go
import shap
import matplotlib.cm as cm

model, feature_names = joblib.load("Heart_Disease_Project/ui/heart_model.pkl")

st.set_page_config(page_title="Heart Check-up", layout="centered")
st.title("AI Heart Disease Prediction")

st.sidebar.header("Inputs Guide")
st.sidebar.info("""
- **Sex:** 1=Male, 0=Female
- **Chest Pain (cp):** 0=Typical, 1=Atypical, 2=Non-anginal, 3=Asymptomatic
- **Thalassemia (thal):** 1=Normal, 2=Fixed defect, 3=Reversible defect
- **Slope:** 0=Upsloping, 1=Flat, 2=Downsloping
- **Exang:** 1=Yes, 0=No (Exercise induced angina)
""")

st.write("### Enter Patient Data:")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex (1: Male, 0: Female)", [1, 0])
    cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
    thalach = st.number_input("Max Heart Rate (thalach)", 40, 220, 150)
    exang = st.selectbox("Exercise Induced Angina (1:Yes, 0:No)", [0, 1])

with col2:
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 0.0)
    ca = st.slider("Major Vessels (0-3)", 0, 3, 0)
    thal = st.selectbox("Thalassemia (1-3)", [1, 2, 3])
    slope = st.selectbox("ST Slope (0-2)", [0, 1, 2])

if st.button("Predict"):
    input_df = pd.DataFrame([{
        "thal": thal, "ca": ca, "slope": slope, "thalach": thalach,
        "exang": exang, "sex": sex, "oldpeak": oldpeak, "cp": cp, "age": age
    }])
    input_df = input_df[feature_names]

    proba = model.predict_proba(input_df)
    percent_sick = proba[0][0] * 100
    percent_safe = proba[0][1] * 100

    if percent_sick > 50:
        st.error(f"Heart Disease Risk: {percent_sick:.2f}%")
        predicted_value = percent_sick
    else:
        st.success(f"Safety Score: {percent_safe:.2f}%")
        predicted_value = percent_sick

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=predicted_value,
        title={'text': "Danger Level (%)"},
        gauge={'axis': {'range': [0, 100]}, 'bar': {'color': "black", 'thickness': 0}}
    ))
    st.plotly_chart(fig_gauge)

    st.write("---")
    st.subheader("Deep Analysis: Why this result?")

    try:
        with st.spinner('Deep calculating... this takes 10 seconds for SVM.'):
            explainer = shap.KernelExplainer(model.predict_proba, input_df.values)
            raw_sv = explainer.shap_values(input_df.values, nsamples=500)

            if isinstance(raw_sv, list):
                sv0 = np.array(raw_sv[0]).flatten()
                sv1 = np.array(raw_sv[1]).flatten() if len(raw_sv) > 1 else sv0
                sv = sv1 if np.abs(sv1).sum() > np.abs(sv0).sum() else sv0
            else:
                sv = np.array(raw_sv).flatten()

            impact_df = pd.DataFrame({'Feature': feature_names, 'Impact': sv}).sort_values(by='Impact')

            display_impact = impact_df['Impact'].values
            if np.abs(display_impact).max() < 0.01 and np.abs(display_impact).max() > 0:
                display_impact = display_impact * 1000
                st.caption("Note: Values scaled x1000 for visibility.")

            fig_impact = px.bar(
                impact_df, x=display_impact, y='Feature', orientation='h',
                color=display_impact,
                color_continuous_scale='RdBu_r', 
                template='plotly_dark'
            )
            
            max_range = max(abs(display_impact).max(), 0.001) * 1.2
            fig_impact.update_layout(xaxis=dict(range=[-max_range, max_range]))
            
            st.plotly_chart(fig_impact, use_container_width=True)

    except Exception as e:
        st.error(f"Analysis error: {e}")
