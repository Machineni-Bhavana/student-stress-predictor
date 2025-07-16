import streamlit as st
import pandas as pd
import joblib

# Load the saved model and scaler
model = joblib.load('stress_model.pkl')
scaler = joblib.load('scaler.pkl')

# Streamlit UI
st.title("Student Stress Level Predictor")

# Input fields for user to enter new data
sleep_quality = st.slider("Sleep Quality (1=Very Poor, 5=Excellent)", 1, 5, 3)
headache_freq = st.slider("Headache Frequency per Week", 0, 7, 2)
academic_perf = st.slider("Academic Performance (1=Very Poor, 5=Excellent)", 1, 5, 3)
study_load = st.slider("Study Load (1=Very Low, 5=Very High)", 1, 5, 3)
extracurricular = st.slider("Extracurricular Activities per Week", 0, 7, 2)

# Prepare input data as DataFrame
input_df = pd.DataFrame({
    'Sleep_Quality': [sleep_quality],
    'Headache_Frequency': [headache_freq],
    'Academic_Performance': [academic_perf],
    'Study_Load': [study_load],
    'Extracurricular_Activities': [extracurricular]
})

# Scale the inputs using the saved scaler
input_scaled = scaler.transform(input_df)

# Predict stress level
pred = model.predict(input_scaled)[0]

# Map prediction to label
stress_map = {0: "Low", 1: "Medium", 2: "High"}
predicted_stress = stress_map[pred]

st.subheader(f"Predicted Stress Level: {predicted_stress}")

# Add actionable recommendations based on prediction
if predicted_stress == "High":
    st.error("⚠️ High stress detected! Try to improve your sleep quality, reduce study load, and include relaxation activities like meditation or light exercise.")
elif predicted_stress == "Medium":
    st.warning("⚠️ Moderate stress. Maintain a balanced routine with regular breaks and some physical activity.")
else:
    st.success("✅ Low stress. Keep up the good habits!")

