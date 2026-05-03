# src/predict.py

import joblib
import pandas as pd

# -----------------------------
# Load trained model
# -----------------------------
MODEL_PATH = "model/fraud_model.pkl"
model = joblib.load(MODEL_PATH)


# -----------------------------
# Prediction Function
# -----------------------------
def predict(input_data: dict, threshold: float = 0.3):
    """
    Predict fraud probability

    Parameters:
    ----------
    input_data : dict
        Example:
        {
            "V1": -1.359807,
            "V2": -0.072781,
            ...
            "V28": -0.021053,
            "Amount": 149.62
        }

    threshold : float
        Probability cutoff (default = 0.3)

    Returns:
    -------
    dict
    """

    try:
        # Convert input to DataFrame
        df = pd.DataFrame([input_data])

        # Ensure all numeric
        df = df.astype(float)

        # Prediction probability
        prob = float(model.predict_proba(df)[0][1])

        # Apply threshold
        prediction = prob > threshold

        return {
            "fraud": bool(prediction),
            "fraud_probability": round(prob, 4),
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

    sample_input = {
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
    }

    result = predict(sample_input)
    print(result)