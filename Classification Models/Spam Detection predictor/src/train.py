# src/train.py

import pandas as pd
import joblib
import re

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import BernoulliNB


def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9 ]', '',text)
    return text

# -----------------------------
# 1. Load Data
# -----------------------------
def load_data(path: str):
    df = pd.read_csv(path)

    # Rename columns if needed
    df.columns = ["Category", "Message"]

    # Encode target
    df["Category"] = df["Category"].map({"ham": 0, "spam": 1})
    
    df["Message"] = df["Message"].apply(clean_text)

    return df


# -----------------------------
# 2. Split Data
# -----------------------------
def split_data(df):
    X = df["Message"]
    y = df["Category"]

    return train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )


# -----------------------------
# 3. Build Pipeline
# -----------------------------
def build_pipeline():
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            binary=True,
            max_features=5000
        )),
        ("model", BernoulliNB())
    ])

    return pipeline


# -----------------------------
# 4. Train & Save
# -----------------------------
def train():
    print("📥 Loading data...")
    df = load_data("data/spam.csv")

    print("🔀 Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("⚙️ Building pipeline...")
    model = build_pipeline()

    print("🚀 Training model...")
    model.fit(X_train, y_train)

    print("💾 Saving model...")
    joblib.dump(model, "model/spam_model.pkl")

    print("✅ Training complete! Model saved at model/spam_model.pkl")


# -----------------------------
# 5. Run
# -----------------------------
if __name__ == "__main__":
    train()