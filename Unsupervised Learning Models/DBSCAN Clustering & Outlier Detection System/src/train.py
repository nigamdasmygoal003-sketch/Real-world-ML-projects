# src/train.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

DATA_PATH = "data/DBSCAN.csv"
MODEL_PATH = "model/dbscan_pipeline.pkl"


def load_data(path):
    return pd.read_csv(path)


def build_pipeline():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("dbscan", DBSCAN(eps=0.2, min_samples=5))
    ])


def train():
    print("📥 Loading data...")
    df = load_data(DATA_PATH)

    X = df[["Weight", "Height"]]

    print("⚙️ Building pipeline...")
    model = build_pipeline()

    print("🏋️ Training...")
    model.fit(X)

    labels = model["dbscan"].labels_
    df["cluster"] = labels

    print("\n📊 Cluster Distribution:")
    print(df["cluster"].value_counts())

    print("\n💾 Saving model...")
    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("✅ Done!")


if __name__ == "__main__":
    train()