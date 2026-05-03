# app.py

import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Load Model
# -----------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

model = load_model()

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="Rain Prediction", layout="wide")

st.title("🌧️ Rain Prediction System")
st.markdown("Predict whether it will rain tomorrow")

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Input Weather Data")

def user_input():

    # Location
    Location = st.sidebar.selectbox(
        "Location",
        ["Albury", "Sydney", "Melbourne", "Brisbane", "Perth"]
    )

    # Temperature
    st.sidebar.subheader("🌡️ Temperature")
    MinTemp = st.sidebar.slider("Min Temp", -5.0, 40.0, 15.0)
    MaxTemp = st.sidebar.slider("Max Temp", -5.0, 50.0, 25.0)
    Temp9am = st.sidebar.slider("Temp 9am", -5.0, 40.0, 20.0)
    Temp3pm = st.sidebar.slider("Temp 3pm", -5.0, 50.0, 25.0)

    # Rain
    st.sidebar.subheader("🌧️ Rain")
    Rainfall = st.sidebar.slider("Rainfall", 0.0, 200.0, 0.0)
    RainToday = st.sidebar.selectbox("Rain Today", [0, 1])

    # Wind
    st.sidebar.subheader("💨 Wind")
    WindGustSpeed = st.sidebar.slider("Wind Gust Speed", 0.0, 150.0, 30.0)
    WindSpeed9am = st.sidebar.slider("Wind Speed 9am", 0.0, 100.0, 20.0)
    WindSpeed3pm = st.sidebar.slider("Wind Speed 3pm", 0.0, 100.0, 20.0)

    WindGustDir = st.sidebar.selectbox("Wind Gust Dir", ["N","NE","E","SE","S","SW","W","NW"])
    WindDir9am = st.sidebar.selectbox("Wind Dir 9am", ["N","NE","E","SE","S","SW","W","NW"])
    WindDir3pm = st.sidebar.selectbox("Wind Dir 3pm", ["N","NE","E","SE","S","SW","W","NW"])

    # Humidity & Pressure
    st.sidebar.subheader("💧 Humidity & Pressure")
    Humidity9am = st.sidebar.slider("Humidity 9am", 0.0, 100.0, 50.0)
    Humidity3pm = st.sidebar.slider("Humidity 3pm", 0.0, 100.0, 50.0)
    Pressure9am = st.sidebar.slider("Pressure 9am", 980.0, 1050.0, 1015.0)
    Pressure3pm = st.sidebar.slider("Pressure 3pm", 980.0, 1050.0, 1015.0)

    # Clouds & Others
    st.sidebar.subheader("☁️ Clouds & Others")
    Cloud9am = st.sidebar.slider("Cloud 9am", 0.0, 9.0, 5.0)
    Cloud3pm = st.sidebar.slider("Cloud 3pm", 0.0, 9.0, 5.0)
    Evaporation = st.sidebar.slider("Evaporation", 0.0, 50.0, 5.0)
    Sunshine = st.sidebar.slider("Sunshine", 0.0, 15.0, 7.0)

    data = {
        "Location": Location,
        "MinTemp": MinTemp,
        "MaxTemp": MaxTemp,
        "Rainfall": Rainfall,
        "Evaporation": Evaporation,
        "Sunshine": Sunshine,
        "WindGustDir": WindGustDir,
        "WindGustSpeed": WindGustSpeed,
        "WindDir9am": WindDir9am,
        "WindDir3pm": WindDir3pm,
        "WindSpeed9am": WindSpeed9am,
        "WindSpeed3pm": WindSpeed3pm,
        "Humidity9am": Humidity9am,
        "Humidity3pm": Humidity3pm,
        "Pressure9am": Pressure9am,
        "Pressure3pm": Pressure3pm,
        "Cloud9am": Cloud9am,
        "Cloud3pm": Cloud3pm,
        "Temp9am": Temp9am,
        "Temp3pm": Temp3pm,
        "RainToday": RainToday
    }

    return pd.DataFrame([data])


input_df = user_input()

# -----------------------------
# Display Input
# -----------------------------
st.subheader("📊 Input Data")
st.dataframe(input_df)

# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict"):

    try:
        prob = model.predict_proba(input_df)[0][1]

        # Threshold tuning
        threshold = 0.3
        prediction = int(prob >= threshold)

        st.subheader("🔮 Prediction")

        if prediction == 1:
            st.error(f"🌧️ Rain Expected\nProbability: {prob:.2f}")
        else:
            st.success(f"☀️ No Rain\nProbability: {prob:.2f}")

        # Risk Level
        st.subheader("⚠️ Risk Level")

        if prob < 0.3:
            st.success("🟢 Low Risk")
        elif prob < 0.6:
            st.warning("🟡 Medium Risk")
        else:
            st.error("🔴 High Risk")

    except Exception as e:
        st.error(f"Error: {e}")