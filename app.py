import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load('diabetes_model.pkl')

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.caption("AI MODEL DEVELOPED BY JAIDEV AND ANMOL")

st.title("🩺 Diabetes Prediction Web Application")
st.write("Enter patient medical parameters below to check for diabetes risk.")

# User inputs layout
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
    glucose = st.number_input("Glucose Level (mg/dL)", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure (mm Hg)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=20)

with col2:
    insulin = st.number_input("Insulin Level (mu U/ml)", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI Value", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5, format="%.3f")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

# Prediction button
if st.button("Predict Diabetes Risk", type="primary"):
    user_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(user_data)
    probability = model.predict_proba(user_data)[0][1] * 100

    st.markdown("---")
    if prediction[0] == 1:
        st.error(f"⚠️ **High Risk of Diabetes Detected** (Confidence: {probability:.1f}%)")
        st.info("Recommendation: Please consult a medical specialist for further diagnostic evaluation.")
    else:
        st.success(f"✅ **Low Risk of Diabetes Detected** (Confidence: {100 - probability:.1f}%)")
        st.info("Recommendation: Maintain a balanced diet and regular physical exercise.")
