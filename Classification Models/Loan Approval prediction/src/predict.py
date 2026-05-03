# src/predict.py

import joblib
import pandas as pd

# Load model once
MODEL_PATH = "model/loan_model.pkl"
model = joblib.load(MODEL_PATH)


def predict(input_data: dict):
    """
    Predict loan approval

    Parameters:
    ----------
    input_data : dict
        Example:
        {
            "city": "Kolkata",
            "income": 60000,
            "credit_score": 720,
            "loan_amount": 250000,
            "years_employed": 4,
            "points": 0.75
        }

    Returns:
    -------
    dict
        {
            "prediction": True/False,
            "approval_probability": float,
            "threshold_used": float
        }
    """

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # Get probability of approval (class = 1)
        probability = float(model.predict_proba(df)[0][1])

        # Business threshold (adjustable)
        threshold = 0.6

        # Final decision
        prediction = probability > threshold

        return {
            "prediction": bool(prediction),
            "approval_probability": round(probability, 3),
            "threshold_used": threshold
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# Test block (run this file directly)
if __name__ == "__main__":
    sample_input = {
        "city": "Delhi",
        "income": 50000,
        "credit_score": 700,
        "loan_amount": 200000,
        "years_employed": 5,
        "points": 0.8
    }

    result = predict(sample_input)
    print(result)