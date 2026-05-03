import streamlit as st
from src.predict import predict

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Loan Approval Predictor", layout="centered")

st.title("🏦 Loan Approval Predictor")
st.markdown("Predict whether a loan will be **approved or rejected**")

# =========================
# INPUT (SIDEBAR)
# =========================
st.sidebar.header("Enter Applicant Details")

city = st.sidebar.text_input("City")
income = st.sidebar.number_input("Income", min_value=0.0)
credit_score = st.sidebar.number_input("Credit Score", min_value=0.0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0.0)
years_employed = st.sidebar.number_input("Years Employed", min_value=0.0)
points = st.sidebar.number_input("Points (0–100)", min_value=0.0, max_value=100.0)

# =========================
# DISPLAY INPUT
# =========================
st.subheader("📌 Input Summary")

input_data = {
    "city": city,
    "income": income,
    "credit_score": credit_score,
    "loan_amount": loan_amount,
    "years_employed": years_employed,
    "points": points
}

st.write(input_data)

# =========================
# PREDICTION
# =========================
if st.button("Predict Loan Approval"):
    try:
        result = predict(input_data)

        if "error" in result:
            st.error(f"⚠️ Error: {result['error']}")
        else:
            prediction = result["prediction"]
            prob = result["approval_probability"]

            if prediction:
                st.success(f"✅ LOAN APPROVED\n\nConfidence: {prob:.2f}")
            else:
                st.error(f"❌ LOAN REJECTED\n\nConfidence: {prob:.2f}")

    except Exception as e:
        st.error(f"⚠️ Error: {str(e)}")