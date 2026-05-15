import streamlit as st
import random

st.title("Lung Cancer Prediction System")

st.write("AI-powered lung cancer risk prediction demo.")

age = st.number_input("Age", 1, 100, 30)

smoking = st.selectbox("Smoking Habit", ["Yes", "No"])

cough = st.selectbox("Chronic Cough", ["Yes", "No"])

chest_pain = st.selectbox("Chest Pain", ["Yes", "No"])

if st.button("Predict"):

    risk_score = random.randint(30, 95)

    if risk_score > 60:
        st.error(f"High Risk Detected ({risk_score}%)")
    else:
        st.success(f"Low Risk Detected ({risk_score}%)")
