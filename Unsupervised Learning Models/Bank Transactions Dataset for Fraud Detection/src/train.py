# src/train.py

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest

# -------------------------
# CONFIG
# -------------------------
DATA_PATH = "data/bank_transactions.csv"
MODEL_PATH = "model/isolation_forest_pipeline.pkl"
CONTAMINATION = 0.02  # ~2% anomalies


# -------------------------
# LOAD DATA
# -------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# -------------------------
# PREPROCESSING + PIPELINE
# -------------------------
def build_pipeline():

    num_features = [
        "TransactionAmount",
        "TransactionDuration",
        "LoginAttempts",
        "AccountBalance",
        "CustomerAge"
    ]

    cat_features = [
        "TransactionType",
        "Channel",
        "Location",
        "CustomerOccupation"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("isolation_forest", IsolationForest(
            n_estimators=100,
            contamination=CONTAMINATION,
            random_state=42
        ))
    ])

    return pipeline


# -------------------------
# DATA CLEANING
# -------------------------
def clean_data(df):

    cols_to_drop = [
        "TransactionID",
        "AccountID",
        "DeviceID",
        "IP Address",
        "MerchantID",
        "TransactionDate"
    ]

    df = df.drop(columns=cols_to_drop)

    return df


# -------------------------
# TRAIN
# -------------------------
def train():

    print("📥 Loading data...")
    df = load_data(DATA_PATH)

    print("🧹 Cleaning data...")
    df = clean_data(df)

    print("⚙️ Building pipeline...")
    model = build_pipeline()

    print("🏋️ Training model...")
    model.fit(df)

    # -------------------------
    # ANOMALY DETECTION
    # -------------------------
    print("\n📊 Detecting anomalies...")

    labels = model.predict(df)
    df["anomaly"] = labels

    # summary
    print("\n📈 Anomaly Distribution:")
    print(df["anomaly"].value_counts())

    # show some anomalies
    print("\n🚨 Sample Anomalies:")
    print(df[df["anomaly"] == -1].head())

    # -------------------------
    # SAVE MODEL
    # -------------------------
    print("\n💾 Saving model...")
    os.makedirs("../model", exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print("✅ Training complete!")
    print(f"Model saved at: {MODEL_PATH}")


# -------------------------
# RUN
# -------------------------
if __name__ == "__main__":
    train()