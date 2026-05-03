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
st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("🏠 California House Price Predictor")
st.markdown("Predict house prices based on housing features")

# =========================
# INPUT (SIDEBAR)
# =========================
st.sidebar.header("Enter House Details")

median_income = st.sidebar.number_input("Median Income", min_value=0.0)
house_age = st.sidebar.number_input("House Age", min_value=0.0)
avg_rooms = st.sidebar.number_input("Average Rooms", min_value=0.0)
avg_bedrooms = st.sidebar.number_input("Average Bedrooms", min_value=0.0)
population = st.sidebar.number_input("Population", min_value=0.0)
avg_occupancy = st.sidebar.number_input("Average Occupancy", min_value=0.0)
latitude = st.sidebar.number_input("Latitude", value=34.0)
longitude = st.sidebar.number_input("Longitude", value=-118.0)

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Summary")

input_data = {
    "Median Income": median_income,
    "House Age": house_age,
    "Avg Rooms": avg_rooms,
    "Avg Bedrooms": avg_bedrooms,
    "Population": population,
    "Avg Occupancy": avg_occupancy,
    "Latitude": latitude,
    "Longitude": longitude
}

st.write(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict Price"):
    try:
        values = [
            median_income,
            house_age,
            avg_rooms,
            avg_bedrooms,
            population,
            avg_occupancy,
            latitude,
            longitude
        ]

        features = np.array([values])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features)[0]
        price = prediction * 1000

        st.success(f"💰 Estimated Price: ${price:,.2f}")

    except:
        st.error("⚠️ Please enter valid numbers!")