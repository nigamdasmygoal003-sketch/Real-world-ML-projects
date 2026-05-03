import streamlit as st
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD SAVED FILES
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Car Price Predictor", layout="wide")

st.title("🚗 Car Price Predictor")
st.markdown("Predict the price of a car based on its features")

# =========================
# INPUT SECTION
# =========================
st.sidebar.header("Enter Car Details")

def user_input():
    # Numerical inputs
    levy = st.sidebar.number_input("Levy", min_value=0.0)
    prod_year = st.sidebar.number_input("Production Year", min_value=1990, max_value=2025)
    engine_volume = st.sidebar.number_input("Engine Volume", min_value=0.0)
    mileage = st.sidebar.number_input("Mileage", min_value=0.0)
    cylinders = st.sidebar.number_input("Cylinders", min_value=1)
    airbags = st.sidebar.number_input("Airbags", min_value=0)
    doors = st.sidebar.number_input("Doors", min_value=2, max_value=6)

    # Categorical inputs
    manufacturer = st.sidebar.selectbox("Manufacturer", ["Toyota","BMW","Mercedes","Ford","Honda"])
    model_name = st.sidebar.selectbox("Model", ["Corolla","X5","C-Class","Focus","Civic"])
    category = st.sidebar.selectbox("Category", ["Sedan","SUV","Hatchback"])
    leather = st.sidebar.selectbox("Leather Interior", ["Yes","No"])
    fuel = st.sidebar.selectbox("Fuel Type", ["Petrol","Diesel","Hybrid"])
    gear = st.sidebar.selectbox("Gear Box Type", ["Manual","Automatic"])
    drive = st.sidebar.selectbox("Drive Wheels", ["FWD","RWD","4x4"])
    wheel = st.sidebar.selectbox("Wheel", ["Left wheel","Right-hand drive"])
    color = st.sidebar.selectbox("Color", ["Black","White","Silver","Blue","Red"])

    data = {
        "Levy": levy,
        "Manufacturer": manufacturer,
        "Model": model_name,
        "Prod. year": prod_year,
        "Category": category,
        "Leather interior": leather,
        "Fuel type": fuel,
        "Engine volume": engine_volume,
        "Mileage": mileage,
        "Cylinders": cylinders,
        "Gear box type": gear,
        "Drive wheels": drive,
        "Doors": doors,
        "Wheel": wheel,
        "Color": color,
        "Airbags": airbags
    }

    return pd.DataFrame([data])

input_df = user_input()

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Data")
st.write(input_df)

# =========================
# PREDICTION
# =========================
if st.button("Predict Price"):
    try:
        # Separate columns
        num_cols = input_df.select_dtypes(include=["int64","float64"]).columns
        cat_cols = input_df.select_dtypes(include=["object"]).columns

        # Preprocessing
        X_num = num_imputer.transform(input_df[num_cols])
        X_num = scaler.transform(X_num)

        X_cat = cat_imputer.transform(input_df[cat_cols])
        X_cat = encoder.transform(X_cat)

        X_final = np.hstack([X_num, X_cat])

        # Prediction
        prediction = model.predict(X_final)[0]

        st.success(f"💰 Estimated Price: ${prediction:,.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")