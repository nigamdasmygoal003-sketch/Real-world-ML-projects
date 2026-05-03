import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODELS
# =========================
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
model = joblib.load("model.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Insurance Cost Predictor", layout="centered")

st.title("💰 Medical Insurance Cost Predictor")
st.markdown("Predict insurance charges based on personal details")

# =========================
# INPUT (SIDEBAR)
# =========================
st.sidebar.header("Enter Details")

age = st.sidebar.number_input("Age", min_value=0)
bmi = st.sidebar.number_input("BMI", min_value=0.0)
children = st.sidebar.number_input("Children", min_value=0)

sex = st.sidebar.selectbox("Sex", ["male", "female"])
smoker = st.sidebar.selectbox("Smoker", ["yes", "no"])
region = st.sidebar.selectbox("Region", ["southwest", "southeast", "northwest", "northeast"])

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Summary")

input_data = {
    "Age": age,
    "BMI": bmi,
    "Children": children,
    "Sex": sex,
    "Smoker": smoker,
    "Region": region
}

st.write(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict Insurance Cost"):
    try:
        # Manual encoding (same as training)
        sex_val = 1 if sex == "male" else 0
        smoker_val = 1 if smoker == "yes" else 0

        # Numerical
        num_data = np.array([[age, bmi, children, sex_val, smoker_val]])

        # Categorical
        cat_data = np.array([[region]])

        # Preprocessing
        num_scaled = scaler.transform(num_data)
        cat_encoded = encoder.transform(cat_data)

        final_input = np.hstack([num_scaled, cat_encoded])

        # Prediction
        prediction = model.predict(final_input)[0]

        st.success(f"💰 Estimated Charges: ₹ {prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")