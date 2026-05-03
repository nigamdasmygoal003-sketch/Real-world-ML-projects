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
st.set_page_config(page_title="Energy Production Predictor", layout="centered")

st.title("⚡ Energy Production Predictor")
st.markdown("Predict energy production based on time and environmental factors")

# =========================
# INPUT (SIDEBAR)
# =========================
st.sidebar.header("Enter Details")

start_hour = st.sidebar.number_input("Start Hour", min_value=0, max_value=23)
end_hour = st.sidebar.number_input("End Hour", min_value=0, max_value=23)
day_of_year = st.sidebar.number_input("Day of Year", min_value=1, max_value=365)

sources = ["Solar", "Wind", "Hydro"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
seasons = ["Winter", "Spring", "Summer", "Fall"]

source = st.sidebar.selectbox("Source", sources)
day = st.sidebar.selectbox("Day Name", days)
month = st.sidebar.selectbox("Month Name", months)
season = st.sidebar.selectbox("Season", seasons)

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Summary")

input_data = {
    "Start Hour": start_hour,
    "End Hour": end_hour,
    "Day of Year": day_of_year,
    "Source": source,
    "Day": day,
    "Month": month,
    "Season": season
}

st.write(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict Production"):
    try:
        # Numerical data
        num_data = np.array([[start_hour, end_hour, day_of_year]])

        # Categorical data
        cat_data = np.array([[source, day, month, season]])

        # Preprocessing
        num_scaled = scaler.transform(num_data)
        cat_encoded = encoder.transform(cat_data)

        final_input = np.hstack([num_scaled, cat_encoded])

        # Prediction
        prediction = model.predict(final_input)[0]

        st.success(f"⚡ Predicted Production: {prediction:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")