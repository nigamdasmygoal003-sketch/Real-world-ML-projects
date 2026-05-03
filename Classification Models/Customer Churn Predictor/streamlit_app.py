import streamlit as st
from src.predict import predict

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Customer Churn Predictor", layout="wide")

st.title("📊 Customer Churn Prediction System")
st.markdown("Predict whether a customer is likely to churn")

# =========================
# SIDEBAR INPUT
# =========================
st.sidebar.header("Customer Details")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
senior = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Partner", ["Yes", "No"])
dependents = st.sidebar.selectbox("Dependents", ["Yes", "No"])

tenure = st.sidebar.number_input("Tenure (months)", min_value=0.0)
monthly = st.sidebar.number_input("Monthly Charges", min_value=0.0)
total = st.sidebar.number_input("Total Charges", min_value=0.0)

phone = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
multiple = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])

internet = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
security = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
backup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
device = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
tech = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
tv = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
movies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])

contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
paperless = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
payment = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# =========================
# INPUT SUMMARY
# =========================
st.subheader("📌 Input Summary")

input_data = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": tenure,
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": security,
    "OnlineBackup": backup,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": tv,
    "StreamingMovies": movies,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": monthly,
    "TotalCharges": total
}

st.write(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict Churn"):
    try:
        result = predict(input_data)

        if "error" in result:
            st.error(result["error"])
        else:
            prob = result["churn_probability"]

            if result["churn"]:
                st.error(f"⚠️ HIGH CHURN RISK\n\nProbability: {prob:.2f}")
            else:
                st.success(f"✅ LOW CHURN RISK\n\nProbability: {prob:.2f}")

    except Exception as e:
        st.error(f"Error: {str(e)}")