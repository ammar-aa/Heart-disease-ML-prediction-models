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


st.set_page_config(page_title="Initial check up for heart diseases", layout="centered")
st.title("ML model predict diseases based on data")
st.write("Enter your data:")


age = st.number_input("Age", 1, 120, 30)
sex = st.selectbox("Sex: Male=1, Female=0", [0,1])
thalach = st.number_input("Maximum heart rate achieved during exercise", 40, 220, 150)
oldpeak = st.number_input("ST depression induced by exercise relative to rest", 0.0, 7.0, 0.0)
ca = st.slider("Number of major vessels (0-3) colored by fluoroscopy", 0, 3, 0)
exang = st.selectbox("Exercise induced angina: 1=Yes, 0=No", [0,1])
cp = st.selectbox("Chest pain type: 0=Typical angina, 1=Atypical angina, 2=Non-anginal pain, 3=Asymptomatic", [0,1,2,3])
thal = st.selectbox("Thalassemia test result: 1=Normal, 2=Fixed defect, 3=Reversible defect", [0,1,2,3])
slope = st.selectbox("Slope of the peak exercise ST segment: 0=Upsloping, 1=Flat, 2=Downsloping", [0,1,2])


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

    
    pred = model.predict(input_df)
    proba = model.predict_proba(input_df)
    percent_safe = proba[0][1] * 100
    percent_sick = proba[0][0] * 100

    if percent_sick > 50:
        st.error(f"You have {percent_sick:.2f}% chance of being a heart patient")
        predicted_value = percent_sick 
    else:
        st.success(f"You are {percent_safe:.2f}% safe from heart diseases")
        predicted_value = percent_sick 
     
    n_colors = 1000
    cmap = cm.get_cmap('jet', n_colors)
    colors = []
    for i in range(n_colors):
        r,g,b,a = cmap(i/n_colors)
        r,g,b = int(r*255), int(g*255), int(b*255)
        colors.append(f'rgb({r},{g},{b})')
    ranges = np.linspace(0, predicted_value, n_colors+1)
    steps = [{'range':[ranges[i], ranges[i+1]], 'color': colors[i]} for i in range(n_colors)]

 
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

    
    st.write("---")
    st.subheader("Deep Analysis: Why this result?")

    try:
        with st.spinner('Calculating impact... This takes a moment for SVM models.'):
            def model_predict(data):
                temp_df = pd.DataFrame(data, columns=feature_names)
                return model.predict_proba(temp_df)

            explainer = shap.KernelExplainer(model_predict, input_df.values)
            
            all_shap_values = explainer.shap_values(input_df.values, nsamples=100)
            
            class_idx = 0 
            
            if isinstance(all_shap_values, list):
                sv = all_shap_values[class_idx]
                bv = explainer.expected_value[class_idx]
            else:
                sv = all_shap_values[:, :, class_idx]
                bv = explainer.expected_value[class_idx]

            st.write("How each factor pushed the probability:")
            
            p = shap.force_plot(
                bv, 
                sv, 
                input_df,
                link="logit" 
            )
            
            shap_html = f"<head>{shap.getjs()}</head><body>{p.html()}</body>"
            components.html(shap_html, height=500)
            
            st.info("""
            **Explanation:**
            - **Red:** Factors pushing towards 'Sick'.
            - **Blue:** Factors pushing towards 'Safe'.
            """)

    except Exception as e:
        st.error(f"Analysis Error: {e}")
