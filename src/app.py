
import streamlit as st
import pandas as pd
import joblib

# Load saved model and feature columns
model = joblib.load("../diabetes_model.pkl")
model_columns = joblib.load("../model_columns.pkl")


# Title and description
st.title("🩺 Diabetes Risk Prediction App")
st.write("Enter patient details below to predict diabetes risk:")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.number_input("Glucose Level", 0, 200, 100)
blood_pressure = st.number_input("Blood Pressure", 0, 122, 70)
skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
insulin = st.number_input("Insulin Level", 0, 846, 79)
bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.number_input("Age", 0, 120, 30)

# Feature engineering (same as training)
bmi_category = (
    "Obese" if bmi >= 30 else
    "Overweight" if bmi >= 25 else
    "Normal" if bmi >= 18.5 else
    "Underweight"
)
glucose_bmi_ratio = glucose / bmi if bmi > 0 else 0

# Build input DataFrame with same structure as training
input_data = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age,
    glucose_bmi_ratio,
    1 if bmi_category=="Normal" else 0,
    1 if bmi_category=="Overweight" else 0,
    1 if bmi_category=="Obese" else 0
]], columns=model_columns)

# Prediction button
if st.button("Predict Diabetes Risk"):
    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Risk of Diabetes (Probability: {prob:.2f})")
    else:
        st.success(f"✅ Low Risk of Diabetes (Probability: {prob:.2f})")
