import streamlit as st
import json
from src.predict import predict

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Fraud Detection System", layout="wide")

st.title("💳 Credit Card Fraud Detection")
st.markdown("Detect whether a transaction is **fraudulent or legitimate**")

# =========================
# TABS (LIKE YOUR UI)
# =========================
tab1, tab2 = st.tabs(["Quick Input", "JSON Input"])

# =========================
# QUICK INPUT TAB
# =========================
with tab1:
    st.subheader("Quick Input")

    amount = st.number_input("Amount", min_value=0.0)

    v1 = st.number_input("V1", value=0.0)
    v2 = st.number_input("V2", value=0.0)
    v3 = st.number_input("V3", value=0.0)
    v4 = st.number_input("V4", value=0.0)

    if st.button("Predict (Quick Input)"):
        try:
            data = {
                "Amount": amount,
                "V1": v1,
                "V2": v2,
                "V3": v3,
                "V4": v4
            }

            # fill missing V5–V28
            for i in range(1, 29):
                key = f"V{i}"
                if key not in data:
                    data[key] = 0.0

            result = predict(data)

            if "error" in result:
                st.error(result["error"])
            else:
                fraud = result["fraud"]
                prob = result["fraud_probability"]

                if fraud:
                    st.error(f"🚨 FRAUD DETECTED\n\nProbability: {prob:.2f}")
                else:
                    st.success(f"✅ NORMAL TRANSACTION\n\nProbability: {prob:.2f}")

        except Exception as e:
            st.error(str(e))

# =========================
# JSON INPUT TAB
# =========================
with tab2:
    st.subheader("JSON Input")

    default_json = """{
  "V1": -1.359807,
  "V2": -0.072781,
  "V3": 2.536347,
  "V4": 1.378155,
  "V5": -0.338321,
  "V6": 0.462388,
  "V7": 0.239599,
  "V8": 0.098698,
  "V9": 0.363787,
  "V10": 0.090794,
  "V11": -0.551600,
  "V12": -0.617801,
  "V13": -0.991390,
  "V14": -0.311169,
  "V15": 1.468177,
  "V16": -0.470401,
  "V17": 0.207971,
  "V18": 0.025791,
  "V19": 0.403993,
  "V20": 0.251412,
  "V21": -0.018307,
  "V22": 0.277838,
  "V23": -0.110474,
  "V24": 0.066928,
  "V25": 0.128539,
  "V26": -0.189115,
  "V27": 0.133558,
  "V28": -0.021053,
  "Amount": 149.62
}"""

    json_input = st.text_area("Enter JSON", default_json, height=300)

    if st.button("Predict (JSON Input)"):
        try:
            data = json.loads(json_input)

            result = predict(data)

            if "error" in result:
                st.error(result["error"])
            else:
                fraud = result["fraud"]
                prob = result["fraud_probability"]

                if fraud:
                    st.error(f"🚨 FRAUD DETECTED\n\nProbability: {prob:.2f}")
                else:
                    st.success(f"✅ NORMAL TRANSACTION\n\nProbability: {prob:.2f}")

        except Exception as e:
            st.error(str(e))