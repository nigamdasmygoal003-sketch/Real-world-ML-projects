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
st.set_page_config(page_title="Delhi Traffic Time Predictor", layout="wide")

st.title("🚗 Delhi Traffic Time Predictor")
st.markdown("Estimate travel time based on traffic conditions")

# =========================
# OPTIONS
# =========================
areas = [
    "Vasant Kunj", "Kalkaji", "Greater Kailash", "Janakpuri",
    "Model Town", "Punjabi Bagh", "Dwarka", "Rohini", "Chandni Chowk"
]

time_of_day_options = ["Morning Peak", "Afternoon", "Evening Peak", "Night"]
day_options = ["Weekday", "Weekend"]
weather_options = ["Clear", "Rain", "Fog"]
traffic_options = ["Low", "Medium", "High"]
road_options = ["Highway", "Main Road", "Inner Road"]

# =========================
# INPUT (SIDEBAR)
# =========================
st.sidebar.header("Enter Trip Details")

start_area = st.sidebar.selectbox("Start Area", areas)
end_area = st.sidebar.selectbox("End Area", areas)

distance = st.sidebar.number_input("Distance (km)", min_value=0.1)
speed = st.sidebar.number_input("Average Speed (km/h)", min_value=1.0)

time_of_day = st.sidebar.selectbox("Time of Day", time_of_day_options)
day_type = st.sidebar.selectbox("Day Type", day_options)
weather = st.sidebar.selectbox("Weather", weather_options)
traffic = st.sidebar.selectbox("Traffic Density", traffic_options)
road = st.sidebar.selectbox("Road Type", road_options)

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Summary")

input_dict = {
    "Start Area": start_area,
    "End Area": end_area,
    "Distance (km)": distance,
    "Speed (km/h)": speed,
    "Time": time_of_day,
    "Day": day_type,
    "Weather": weather,
    "Traffic": traffic,
    "Road": road
}

st.write(input_dict)

# =========================
# PREDICTION
# =========================
if st.button("Predict Travel Time"):
    try:
        # Numerical
        num_data = np.array([[distance, speed]])

        # Categorical
        cat_data = np.array([[ 
            start_area,
            end_area,
            time_of_day,
            day_type,
            weather,
            traffic,
            road
        ]])

        # Preprocessing
        num_scaled = scaler.transform(num_data)
        cat_encoded = encoder.transform(cat_data)

        final_input = np.hstack([num_scaled, cat_encoded])

        # Prediction
        prediction = model.predict(final_input)[0]

        st.success(f"⏱ Estimated Travel Time: {prediction:.2f} minutes")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")