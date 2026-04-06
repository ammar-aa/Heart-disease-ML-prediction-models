import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.colors as pc
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go

# Loading model
model, feature_names = joblib.load("Heart_Disease_Project\ui\heart_model.pkl")

# User interface 
st.set_page_config(page_title="Initial check up for heart diseases", layout="centered")
st.title("ML model predict diseases based on data")
st.write("Enter your data:")

# Patient data
age = st.number_input("Age", 35, 110, 50)
sex = st.selectbox("Sex: Male=1, Female=0", [0,1])
thalach = st.number_input("Maximum heart rate achieved during exercise", 60, 220, 120)
oldpeak = st.number_input("ST depression induced by exercise relative to rest", 0.0, 6.5, 1.0)
ca = st.slider("Number of major vessels (0-3) colored by fluoroscopy", 0, 3, 0)
exang = st.selectbox("Exercise induced angina: 1=Yes, 0=No", [0,1])
cp = st.selectbox("Chest pain type: 0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic", [0,1,2,3])
thal = st.selectbox("Thalassemia test result: 1=Normal, 2=Fixed defect, 3=Reversible defect", [0,1,2,3])
slope = st.selectbox("Slope of the peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping", [0,1,2])

# Prediction button
if st.button("predict"):
    input_df = pd.DataFrame([{
        "thal": thal,
        "ca": ca, 
        "slope": slope,
        "thalach": thalach,
        "exang": exang,
        "sex": sex,
        "oldpeak": oldpeak,
        "cp": cp,
        "age": age
    }])
    input_df = input_df[feature_names]

    # Prediction
    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)
    predicted_value = proba[0][1]*100   

    if pred[0] == 1:
        st.error(f" You have {predicted_value:.2f}% chance of being heart patient")
    else:
        st.success(f" You are {proba[0][0]*100:.2f}% safe from heart diseases")

    # gauge meter for sickness percentage
    n_colors = 1000
    cmap = cm.get_cmap('jet', n_colors)
    colors = []
    for i in range(n_colors):
        r,g,b,a = cmap(i/n_colors)
        r,g,b = int(r*255), int(g*255), int(b*255)
        colors.append(f'rgb({r},{g},{b})')
    ranges = np.linspace(0, predicted_value, n_colors+1)
    steps = [{'range':[ranges[i], ranges[i+1]], 'color': colors[i]} for i in range(n_colors)]

    # إنشاء الـ Gauge
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=predicted_value,
        title={'text': "Danger of getting sick (%)"},
        gauge={
            'axis': {'range':[0,100], 'tickcolor':'white'},
            'bgcolor':'black',
            'bar': {'color':'black','thickness':0},
            'steps': steps,
            'borderwidth': 0 
        }
    ))

    st.plotly_chart(fig_gauge)

