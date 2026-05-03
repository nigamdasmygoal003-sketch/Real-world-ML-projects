import os
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans


DATA_PATH = "data/mental_health_digital_behavior_data.csv"
MODEL_PATH = "model/model.pkl"
N_CLUSTERS = 3


def load_data(path):
    df = pd.read_csv(path)
    return df

def build_pipeline(df):
    
    num_features = df.select_dtypes(include=["int64","float64"]).columns
    
    preprocessor = ColumnTransformer([
        ("num",StandardScaler(),num_features)
    ])
    
    pipe = Pipeline([
        ("preprocessor",preprocessor),
        ("KMeans",KMeans(n_clusters=N_CLUSTERS,random_state=42))
    ])
    
    
    return pipe


def train():
    print("📥 Loading data...")
    df = load_data(DATA_PATH)
    
    x = df
    
    print("⚙️ Building pipeline...")
    pipe = build_pipeline(x)
    
    print("🏋️ Training model...")
    pipe.fit(x)
    
    print("\n📊 Cluster Analysis:")
    
    cluster = pipe.predict(x)
    df["cluster"] = cluster
    analysis = df.groupby("cluster").mean()
    print(analysis)
    
    
    print("\n💾 Saving model...")
    os.makedirs("../model", exist_ok=True)
    joblib.dump(pipe, MODEL_PATH)

    print("✅ Training complete!")
    print(f"Model saved at: {MODEL_PATH}")

    
if __name__ == "__main__":
    train()
    
