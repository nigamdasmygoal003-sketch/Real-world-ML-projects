# src/train.py

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(path: str):
    df = pd.read_csv(path)

    # Drop Time (not useful in most cases)
    df = df.drop(columns=["Time"])

    return df


# -----------------------------
# 2. Split Data
# -----------------------------
def split_data(df):
    X = df.drop("Class", axis=1)
    y = df["Class"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # 🔥 VERY IMPORTANT
    )


# -----------------------------
# 3. Build Pipeline
# -----------------------------
def build_pipeline(X):

    num_features = X.columns  # all numeric

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features)
    ])

    pipeline = ImbPipeline([
        ("preprocessor", preprocessor),
        ("smote", SMOTE(random_state=42)),
        ("model", RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42
        ))
    ])

    return pipeline


# -----------------------------
# 4. Train & Save
# -----------------------------
def train():
    print("📥 Loading data...")
    df = load_data("data/creditcard.csv")

    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("⚙️ Building pipeline...")
    model = build_pipeline(X_train)

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    print("💾 Saving model...")
    joblib.dump(model, "model/fraud_model.pkl")

    print("✅ Training complete! Model saved at model/fraud_model.pkl")


# -----------------------------
# 5. Run
# -----------------------------
if __name__ == "__main__":
    train()