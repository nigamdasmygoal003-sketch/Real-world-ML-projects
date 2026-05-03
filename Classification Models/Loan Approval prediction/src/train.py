# src/train.py

import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier


def load_data(path: str):
    df = pd.read_csv(path)

    # Drop useless column
    df = df.drop(columns=["name"])

    return df


def build_pipeline(X):
    num_features = X.select_dtypes(include=["int64", "float64"]).columns
    cat_features = X.select_dtypes(include=["object"]).columns

    # Numerical pipeline
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
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
        ("model", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            random_state=42
        ))
    ])

    return pipeline


def train():
    # Load data
    df = load_data("data/loan_approval.csv")

    # Split features/target
    X = df.drop("loan_approved", axis=1)
    y = df["loan_approved"]

    # Build pipeline
    model = build_pipeline(X)

    # Train on full data
    model.fit(X, y)

    # Save model
    joblib.dump(model, "model/loan_model.pkl")

    print("✅ Model trained and saved successfully!")


if __name__ == "__main__":
    train()