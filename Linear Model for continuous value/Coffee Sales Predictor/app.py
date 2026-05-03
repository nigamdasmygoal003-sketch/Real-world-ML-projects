import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Model
# -------------------------------
@st.cache_resource
def load_model():
    return joblib.load("model/model.pkl")

model = load_model()

# -------------------------------
# UI Config
# -------------------------------
st.set_page_config(page_title="Coffee Sales Predictor", layout="centered")

st.title("☕ Coffee Sales Prediction System")
st.markdown("Predict coffee sales based on time and product features")

# -------------------------------
# Input Section
# -------------------------------
st.subheader("📥 Enter Input Features")

hour_of_day = st.slider("Hour of Day", 0, 23, 10)

coffee_name = st.selectbox(
    "Coffee Type",
    ["Latte", "Americano", "Cappuccino", "Hot Chocolate", "Cocoa"]
)

time_of_day = st.selectbox(
    "Time of Day",
    ["Morning", "Afternoon", "Evening", "Night"]
)

weekday = st.selectbox(
    "Weekday",
    ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
)

month_name = st.selectbox(
    "Month",
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

# Optional numeric features
weekday_sort = st.slider("Weekday Sort (1=Mon, 7=Sun)", 1, 7, 5)
month_sort = st.slider("Month Sort (1-12)", 1, 12, 3)

# -------------------------------
# Prediction
# -------------------------------
if st.button("🔮 Predict Sales"):
    
    input_data = pd.DataFrame([{
        "hour_of_day": hour_of_day,
        "coffee_name": coffee_name,
        "Time_of_Day": time_of_day,
        "Weekday": weekday,
        "Month_name": month_name,
        "Weekdaysort": weekday_sort,
        "Monthsort": month_sort
    }])

    prediction = model.predict(input_data)[0]

    st.success(f"💰 Predicted Sales: ₹ {round(prediction, 2)}")

    # Extra info
    st.info("Model: RandomForestRegressor")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.caption("Built with Streamlit & Scikit-learn")