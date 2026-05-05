import streamlit as st
import numpy as np
import joblib

# Load models
nb_model = joblib.load('models/nb_model.pkl')
ann_model = joblib.load('models/ann_model.pkl')
scaler = joblib.load('models/scaler.pkl')

st.set_page_config(page_title="Medical Diagnosis App")

st.title("🧠 Hybrid Medical Diagnosis System")
st.write("Enter patient details to predict diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
bp = st.number_input("Blood Pressure", 0, 150)
skin = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin, insulin, bmi, dpf, age]])
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Model predictions
    nb_prob = nb_model.predict_proba(input_scaled)[0][1]
    ann_prob = ann_model.predict_proba(input_scaled)[0][1]
    
    # Hybrid combination
    final_prob = 0.4 * nb_prob + 0.6 * ann_prob
    
    result = "Diabetic" if final_prob >= 0.5 else "Non-Diabetic"
    
    st.subheader(f"Prediction: {result}")
    st.write(f"Confidence: {final_prob:.2f}")
    
    # Progress bar
    st.progress(int(final_prob * 100))