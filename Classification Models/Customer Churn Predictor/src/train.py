# src/train.py

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression


# -----------------------------
# 1. Load & Clean Data
# -----------------------------
def load_data(path: str):
    df = pd.read_csv(path)

    # Fix TotalCharges (string → numeric)
    df = df.drop(columns=["customerID"])
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

    # Fill missing values
    df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

    # Convert target
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    # Feature Engineering
    df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

    return df


# -----------------------------
# 2. Build Pipeline
# -----------------------------
def build_pipeline(X):
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical pipeline
    cat_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    # Combine
    preprocessor = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", cat_pipeline, cat_features)
    ])

    # Final pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    return pipeline


# -----------------------------
# 3. Train & Save
# -----------------------------
def train():
    print("Loading data...")
    df = load_data("data/customer_churn.csv")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    print("Building pipeline...")
    model = build_pipeline(X)

    print("Training model...")
    model.fit(X, y)

    print("Saving model...")
    joblib.dump(model, "model/churn_model.pkl")

    print("✅ Training complete! Model saved at model/churn_model.pkl")


# -----------------------------
# 4. Run
# -----------------------------
if __name__ == "__main__":
    train()