import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    st.error("Error loading model or scaler!")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Student Performance Predictor", layout="centered")

st.title("📚 Student Performance Predictor")
st.markdown("Estimate student performance based on study habits")

# =========================
# INPUT (SIDEBAR)
# =========================
st.sidebar.header("Enter Details")

hours_studied = st.sidebar.number_input("Hours Studied", min_value=0.0)
previous_scores = st.sidebar.number_input("Previous Scores", min_value=0.0)
extracurricular = st.sidebar.number_input("Extracurricular Activities (0/1)", min_value=0.0)
sleep_hours = st.sidebar.number_input("Sleep Hours", min_value=0.0)
papers_practiced = st.sidebar.number_input("Sample Papers Practiced", min_value=0.0)

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Summary")

input_data = {
    "Hours Studied": hours_studied,
    "Previous Scores": previous_scores,
    "Extracurricular": extracurricular,
    "Sleep Hours": sleep_hours,
    "Papers Practiced": papers_practiced
}

st.write(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict Performance"):
    try:
        values = [
            hours_studied,
            previous_scores,
            extracurricular,
            sleep_hours,
            papers_practiced
        ]

        features = np.array([values])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)[0]

        st.success(f"📊 Estimated Student Performance: {prediction:.2f}")

    except:
        st.error("⚠️ Please enter valid numbers!")