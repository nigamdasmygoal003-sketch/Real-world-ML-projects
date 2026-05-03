# src/predict.py

import joblib

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "model/spam_model.pkl"
model = joblib.load(MODEL_PATH)


# -----------------------------
# Prediction Function
# -----------------------------
def predict(message: str):
    """
    Predict whether a message is spam or not

    Parameters:
    ----------
    message : str

    Returns:
    -------
    dict
    """

    try:
        # Predict probability
        prob = float(model.predict_proba([message])[0][1])

        # Default threshold = 0.5
        prediction = prob > 0.5

        return {
            "spam": bool(prediction),
            "spam_probability": round(prob, 4)
        }

    except Exception as e:
        return {
            "error": str(e)
        }


# -----------------------------
# Test Block
# -----------------------------
if __name__ == "__main__":

    sample = "WIN a FREE iPhone now!!! Click here to claim your prize!!!"

    result = predict(sample)
    print(result)