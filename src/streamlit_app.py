import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="centered")
st.title("🩺 Diabetes Risk Predictor")

st.markdown("""**Instructions**
1. Train and save the model from the notebook first (creates `diabetes_risk_pipeline.joblib` in project root).
2. Then run this app with: `streamlit run src/streamlit_app.py`
3. Enter patient details in the sidebar and view predicted probability.
""")

with st.sidebar:
    st.header("Patient Inputs")
    Pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    Glucose = st.number_input("Glucose", min_value=0, max_value=300, value=145)
    BloodPressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
    SkinThickness = st.number_input("Skin Thickness", min_value=0, max_value=99, value=25)
    Insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
    BMI = st.number_input("BMI", min_value=0.0, max_value=70.0, value=31.2, step=0.1)
    DPF = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.35, step=0.01)
    Age = st.number_input("Age", min_value=1, max_value=120, value=45)

    # Derived fields must match training logic
    if BMI < 18.5:
        BMI_Category = "Underweight"
    elif BMI < 25:
        BMI_Category = "Normal"
    elif BMI < 30:
        BMI_Category = "Overweight"
    else:
        BMI_Category = "Obese"

    Glucose_BMI_Ratio = (Glucose / BMI) if BMI != 0 else 0.0

    if st.button("Predict Risk"):
        try:
            pipe = joblib.load("diabetes_risk_pipeline.joblib")
            row = pd.DataFrame([{
                "Pregnancies": Pregnancies,
                "Glucose": Glucose,
                "BloodPressure": BloodPressure,
                "SkinThickness": SkinThickness,
                "Insulin": Insulin,
                "BMI": BMI,
                "DiabetesPedigreeFunction": DPF,
                "Age": Age,
                "BMI_Category": BMI_Category,
                "Glucose_BMI_Ratio": Glucose_BMI_Ratio
            }])
            prob = float(pipe.predict_proba(row)[:,1][0])
            pred = int(pipe.predict(row)[0])
            st.success(f"Predicted Probability of Diabetes: {prob:.3f}")
            st.info(f"Predicted Class: {'Diabetic (1)' if pred==1 else 'Not Diabetic (0)'}")
        except FileNotFoundError:
            st.error("Model file not found. Please run the notebook to train and save the model first.")
