import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.colors as pc
import matplotlib.cm as cm
import numpy as np
import plotly.graph_objects as go
import shap
import streamlit.components.v1 as components

model, feature_names = joblib.load("Heart_Disease_Project/ui/heart_model.pkl")

st.set_page_config(page_title="Heart Disease Diagnostic Tool", layout="centered")
st.title("AI Heart Disease Prediction & Analysis")
st.write("Enter patient data to see the risk assessment and key factors:")

col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age", 1, 120, 30)
    sex = st.selectbox("Sex: Male=1, Female=0", [0,1])
    thalach = st.number_input("Max Heart Rate (thalach)", 40, 220, 150)
    oldpeak = st.number_input("ST Depression (oldpeak)", 0.0, 7.0, 0.0)

with col2:
    ca = st.slider("Major Vessels (ca)", 0, 3, 0)
    exang = st.selectbox("Exercise Angina: 1=Yes, 0=No", [0,1])
    cp = st.selectbox("Chest Pain Type (0-3)", [0,1,2,3])
    thal = st.selectbox("Thalassemia (1-3)", [0,1,2,3])

slope = st.selectbox("ST Segment Slope (0-2)", [0,1,2])

if st.button("Predict & Analyze"):
    input_df = pd.DataFrame([{
        "thal": thal, "ca": ca, "slope": slope, "thalach": thalach,
        "exang": exang, "sex": sex, "oldpeak": oldpeak, "cp": cp, "age": age
    }])
    input_df = input_df[feature_names]

    proba = model.predict_proba(input_df)
    percent_safe = proba[0][1] * 100
    percent_sick = proba[0][0] * 100

    if percent_sick > 50:
        st.error(f"High Risk: {percent_sick:.2f}% probability of heart disease.")
        predicted_value = percent_sick 
    else:
        st.success(f"Low Risk: {percent_safe:.2f}% safe from heart disease.")
        predicted_value = percent_sick 
     
    n_colors = 1000
    cmap = cm.get_cmap('jet', n_colors)
    colors = [f'rgb({int(cmap(i/n_colors)[0]*255)},{int(cmap(i/n_colors)[1]*255)},{int(cmap(i/n_colors)[2]*255)})' for i in range(n_colors)]
    ranges = np.linspace(0, predicted_value, n_colors+1)
    steps = [{'range':[ranges[i], ranges[i+1]], 'color': colors[i]} for i in range(n_colors)]

    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=predicted_value,
        title={'text': "Danger Score (%)"},
        gauge={'axis': {'range':[0,100]}, 'bar': {'color':'black','thickness':0}, 'steps': steps}
    ))
    st.plotly_chart(fig_gauge)

    st.write("---")
    st.subheader("Deep Analysis: Why this result?")

    try:
        with st.spinner('Calculating impact of each factor...'):
            def model_predict(data):
                return model.predict_proba(pd.DataFrame(data, columns=feature_names))

            explainer = shap.KernelExplainer(model_predict, input_df.values)
            raw_sv = explainer.shap_values(input_df.values, nsamples=50)

            if isinstance(raw_sv, list):
                sv = np.array(raw_sv[0]).flatten()
            else:
                sv = np.array(raw_sv).flatten()
            
            sv = np.nan_to_num(sv)

            impact_df = pd.DataFrame({
                'Feature': feature_names,
                'Impact': sv
            }).sort_values(by='Impact', ascending=True)

            max_val = max(abs(impact_df['Impact'].max()), abs(impact_df['Impact'].min()))
            limit = max_val * 1.3 if max_val > 0 else 0.1

            fig_impact = px.bar(
                impact_df, x='Impact', y='Feature', orientation='h',
                color='Impact',
                color_continuous_scale=['#0000ff', '#ffffff', '#ff0000'],
                color_continuous_midpoint=0,
                range_x=[-limit, limit], 
                labels={'Impact': 'Strength of Influence'},
                template='plotly_dark'
            )

            fig_impact.update_layout(
                showlegend=False, height=500,
                xaxis=dict(tickformat=".4f", zeroline=True, zerolinewidth=2, zerolinecolor='White'),
                margin=dict(l=20, r=20, t=30, b=20)
            )

            fig_impact.update_traces(
                texttemplate='%{x:.4f}', textposition='outside', cliponaxis=False
            )
            
            st.plotly_chart(fig_impact, use_container_width=True)
            
            st.info("""
            **Reading the Chart:**
            - **Red (Right):** Increases risk.
            - **Blue (Left):** Decreases risk (Protective).
            """)

    except Exception as e:
        st.warning("Visual breakdown unavailable. Prediction remains accurate.")
