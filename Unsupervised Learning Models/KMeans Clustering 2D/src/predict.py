# src/predict.py

import joblib
import numpy as np

MODEL_PATH = "model/kmeans_model.pkl"

# Optional: map cluster → meaning
SEGMENT_MAP = {
    0: "Low Value",
    1: "Medium Value",
    2: "High Value"
}


def load_model(path):
    model = joblib.load(path)
    return model


def predict(model, x, y):
    data = np.array([[x, y]])
    cluster = model.predict(data)[0]

    segment = SEGMENT_MAP.get(cluster, "Unknown")

    return {
        "cluster": int(cluster),
        "segment": segment
    }


if __name__ == "__main__":
    print("🔮 Loading model...")
    model = load_model(MODEL_PATH)

    # Example input (you can change this)
    x = float(input("Enter X value: "))
    y = float(input("Enter Y value: "))

    result = predict(model, x, y)

    print("\n📊 Prediction Result:")
    print(result)