# src/train.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data/Mall_Customers.csv"
MODEL_PATH = "model/kmeans_pipeline.pkl"
N_CLUSTERS = 6


# -------------------------
# LOAD DATA
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# -------------------------
# BUILD PIPELINE
# -------------------------
def build_pipeline():

    numeric_features = ["Age", "Annual Income (k$)", "Spending Score (1-100)"]
    categorical_features = ["Gender"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(drop="first"), categorical_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("kmeans", KMeans(n_clusters=N_CLUSTERS, random_state=42))
    ])

    return pipeline


# -------------------------
# TRAIN FUNCTION
# -------------------------
def train():

    print("📥 Loading data...")
    df = load_data(DATA_PATH)

    X = df.drop(columns=["CustomerID"])

    print("⚙️ Building pipeline...")
    pipeline = build_pipeline()

    print("🏋️ Training model...")
    pipeline.fit(X)

    # -------------------------
    # CLUSTER ANALYSIS (IMPORTANT)
    # -------------------------
    print("\n📊 Cluster Analysis:")

    clusters = pipeline.predict(X)
    df["Cluster"] = clusters

    analysis = df.drop(columns=["CustomerID"]).groupby("Cluster").mean(numeric_only=True)
    print(analysis)

    # -------------------------
    # SAVE MODEL
    # -------------------------
    print("\n💾 Saving model...")
    os.makedirs("../model", exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    print("✅ Training complete!")
    print(f"Model saved at: {MODEL_PATH}")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    train()