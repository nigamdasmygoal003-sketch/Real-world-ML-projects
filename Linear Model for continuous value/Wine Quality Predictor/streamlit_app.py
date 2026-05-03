import streamlit as st
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("model.pkl")
except:
    st.error("Error loading model!")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Wine Quality Predictor", layout="wide")

st.title("🍷 Wine Quality Predictor")
st.markdown("Predict whether a wine is **Good** or **Bad** based on chemical properties")

# =========================
# INPUT (SIDEBAR)
# =========================
st.sidebar.header("Enter Wine Features")

columns = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH",
    "sulphates", "alcohol"
]

inputs = {}

for col in columns:
    inputs[col] = st.sidebar.number_input(col.title(), min_value=0.0)

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Summary")
st.write(inputs)

# =========================
# PREDICTION
# =========================
if st.button("Predict Wine Quality"):
    try:
        values = [inputs[col] for col in columns]
        input_array = np.array([values])

        prediction = model.predict(input_array)[0]

        # OPTIONAL: probability (if model supports it)
        prob = None
        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(input_array)[0][1]

        if prediction == 1:
            st.success("🍷 Good Quality Wine")
        else:
            st.error("⚠️ Bad Quality Wine")

        if prob is not None:
            st.info(f"Confidence: {prob:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")