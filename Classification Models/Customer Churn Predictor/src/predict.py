# src/predict.py

import joblib
import pandas as pd

# -----------------------------
# Load model
# -----------------------------
MODEL_PATH = "model/churn_model.pkl"
model = joblib.load(MODEL_PATH)


# -----------------------------
# Prediction Function
# -----------------------------
def predict(input_data: dict, threshold: float = 0.4):
    """
    Predict customer churn

    Parameters:
    ----------
    input_data : dict
        Example:
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "Yes",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 70.5,
            "TotalCharges": 845.5
        }

    threshold : float
        Probability cutoff (default = 0.4)

    Returns:
    -------
    dict
    """

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # ---- Feature Engineering (must match training!) ----
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

        df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

        # ---- Prediction ----
        prob = float(model.predict_proba(df)[0][1])

        prediction = prob > threshold

        return {
            "churn": bool(prediction),
            "churn_probability": round(prob, 3),
            "threshold_used": threshold
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# -----------------------------
# Test Block
# -----------------------------
if __name__ == "__main__":
    sample = {
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 5,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 75.0,
        "TotalCharges": 350.0
    }

    result = predict(sample)
    print(result)