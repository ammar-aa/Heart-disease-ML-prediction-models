import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go

model, feature_names = joblib.load("Heart_Disease_Project/ui/heart_model.pkl")

st.set_page_config(page_title="Heart Disease Check-up", layout="centered")
st.title("❤️ AI Heart Disease Prediction")
st.write("Enter the patient data to analyze risk level:")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age",0 , 110, 50)
    sex = st.selectbox("Sex (Male=1, Female=0)", [1, 0])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 0, 1000, 120)
    thalach = st.number_input("Max Heart Rate achieved during exercise", 0, 1000, 120)
    cp = st.selectbox("Chest Pain Type (0:Typical angina,1:Atypical angina,2:Non-anginal pain,3:Asymptomatic)",[0, 1, 2, 3])

with col2:
    oldpeak = st.number_input("ST depression induced by exercise relative to rest", 0.0, 6.5, 1.0)
    ca = st.slider("Major Vessels (ca)", 0, 3, 0)
    exang = st.selectbox("Exercise Induced Angina (1=Yes, 0=No)", [0, 1])
    thal = st.selectbox("Thalassemia(1:Normal,2:Fixed defect,3:Reversible defect)", [1, 2, 3])
    slope = st.selectbox("Slope of the peak exercise ST segment(0:Upsloping,1:Flat,2:Downsloping)", [0, 1, 2])


if st.button("Predict & Analyze"):
    
    input_df = pd.DataFrame([{
        "exang": exang, "trestbps": trestbps, "slope": slope, "age": age,
        "thalach": thalach, "thal": thal, "sex": sex, "oldpeak": oldpeak,
        "cp": cp, "ca": ca, 
    }])[feature_names]

    
   
    proba = model.predict_proba(input_df)
    predicted_value = float(proba[0, 0] * 100) 
    if predicted_value > 50:
        st.error(f"⚠️ High Risk: {predicted_value:.2f}% chance of heart disease")
    else:
        st.success(f"✅ Low Risk: {100 - predicted_value:.2f}% Safe")


    n_colors = 100
    cmap = cm.get_cmap('jet', n_colors)
    colors = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b, a in [cmap(i/n_colors) for i in range(n_colors)]]
    
    ranges = np.linspace(0, predicted_value, n_colors+1)
    steps = [{'range': [ranges[i], ranges[i+1]], 'color': colors[i]} for i in range(n_colors)]

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_value,
        number={'suffix': "%", 'font': {'size': 60}},
        title={'text': "Danger Level", 'font': {'size': 20}},
        gauge={
            'axis': {'range': [0, 100], 'tickcolor': 'white'},
            'bar': {'color': 'black', 'thickness': 0},
            'steps': steps,
            'bgcolor': 'black'
        }
    ))
    fig_gauge.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=400)
    st.plotly_chart(fig_gauge)

    st.write("---")
    st.subheader("🔍 Feature Importance Analysis")
    try:
        importances = model.feature_importances_
        feat_importances = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=True)

        fig_importance = px.bar(
            feat_importances, x='Importance', y='Feature', orientation='h',
            color='Importance', color_continuous_scale='RdYlGn_r',
            template='plotly_dark',
            labels={'Importance': 'Impact Strength', 'Feature': 'Medical Test'}
        )
        fig_importance.update_layout(xaxis_title="Relative Impact", yaxis_title="", height=400)
        st.plotly_chart(fig_importance, use_container_width=True)

        top_factor = feat_importances.iloc[-1]['Feature']
        st.info(f"💡 The model found that **{top_factor}** was the most decisive factor for this result.")
    except Exception as e:
        st.warning("Feature importance is only available for Random Forest models.")
