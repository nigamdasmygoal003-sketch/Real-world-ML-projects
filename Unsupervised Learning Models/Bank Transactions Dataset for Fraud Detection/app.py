# app.py

import streamlit as st
import joblib
import pandas as pd

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model/isolation_forest_pipeline.pkl"

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# -------------------------
# UI CONFIG
# -------------------------
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("🏦 Bank Transaction Fraud Detection")
st.write("Detect suspicious transactions using Isolation Forest")

# -------------------------
# INPUT FORM
# -------------------------
st.sidebar.header("📝 Transaction Details")

amount = st.sidebar.number_input("Transaction Amount", min_value=0.0, value=100.0)
duration = st.sidebar.number_input("Transaction Duration (sec)", min_value=1, value=60)
login_attempts = st.sidebar.slider("Login Attempts", 1, 10, 1)
balance = st.sidebar.number_input("Account Balance", min_value=0.0, value=5000.0)
age = st.sidebar.slider("Customer Age", 18, 90, 30)

transaction_type = st.sidebar.selectbox("Transaction Type", ["Debit", "Credit"])
channel = st.sidebar.selectbox("Channel", ["ATM", "Online", "Branch"])
location = st.sidebar.text_input("Location", "New York")
occupation = st.sidebar.selectbox("Customer Occupation", ["Student", "Engineer", "Doctor", "Retired"])

# -------------------------
# INPUT DATAFRAME
# -------------------------
input_df = pd.DataFrame([{
    "TransactionAmount": amount,
    "TransactionDuration": duration,
    "LoginAttempts": login_attempts,
    "AccountBalance": balance,
    "CustomerAge": age,
    "TransactionType": transaction_type,
    "Channel": channel,
    "Location": location,
    "CustomerOccupation": occupation
}])

# -------------------------
# PREDICTION
# -------------------------
prediction = model.predict(input_df)[0]

# -------------------------
# OUTPUT
# -------------------------
st.subheader("🔮 Prediction Result")

if prediction == -1:
    st.error("⚠️ Suspicious Transaction Detected (Anomaly)")
else:
    st.success("✅ Normal Transaction")

# -------------------------
# RISK INSIGHT (Simple logic)
# -------------------------
st.subheader("📊 Risk Insights")

risk_flags = []

if amount > 500:
    risk_flags.append("High transaction amount")

if login_attempts > 3:
    risk_flags.append("Multiple login attempts")

if duration > 200:
    risk_flags.append("Unusually long transaction time")

if age > 70 and channel == "Online":
    risk_flags.append("Unusual digital usage for age group")

if risk_flags:
    for flag in risk_flags:
        st.warning(f"⚠️ {flag}")
else:
    st.info("No strong risk indicators detected")

# -------------------------
# RAW DATA VIEW
# -------------------------
with st.expander("📄 View Input Data"):
    st.dataframe(input_df)