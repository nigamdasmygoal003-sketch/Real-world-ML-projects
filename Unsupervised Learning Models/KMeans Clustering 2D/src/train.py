# src/train.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "data/dataset.csv"
MODEL_PATH = "model/kmeans_model.pkl"
N_CLUSTERS = 3


def load_data(path):
    data = pd.read_csv(path)
    return data


def build_pipeline(n_clusters):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
    ])
    return pipeline


def train():
    print("📥 Loading data...")
    data = load_data(DATA_PATH)

    print("⚙️ Building pipeline...")
    pipeline = build_pipeline(N_CLUSTERS)

    print("🏋️ Training model...")
    pipeline.fit(data)

    print("💾 Saving model...")
    os.makedirs("../model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("✅ Training complete!")
    print(f"Model saved at: {MODEL_PATH}")


if __name__ == "__main__":
    train()