# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model/kmeans_pipeline.pkl"
DATA_PATH = "data/Mall_Customers.csv"

SEGMENT_MAP = {
    0: "Segment 0",
    1: "Segment 1",
    2: "Segment 2",
    3: "Segment 3",
    4: "Segment 4",
    5: "Segment 5"
}

# -------------------------
# LOAD MODEL + DATA
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
df = load_data()

X = df.drop(columns=["CustomerID"])

preprocessor = model["preprocessor"]
kmeans = model["kmeans"]

# -------------------------
# PCA PREP
# -------------------------
X_processed = preprocessor.transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

labels = model.predict(X)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Mall Customer Segmentation", layout="wide")

st.title("🛍️ Mall Customer Segmentation")
st.write("Predict customer segment and visualize it using PCA")

# -------------------------
# INPUTS
# -------------------------
st.sidebar.header("Customer Input")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.slider("Age", 10, 80, 25)
income = st.sidebar.slider("Annual Income (k$)", 10, 150, 50)
score = st.sidebar.slider("Spending Score (1-100)", 1, 100, 50)

# -------------------------
# PREDICTION
# -------------------------
input_df = pd.DataFrame([{
    "Gender": gender,
    "Age": age,
    "Annual Income (k$)": income,
    "Spending Score (1-100)": score
}])

cluster = model.predict(input_df)[0]
segment = SEGMENT_MAP.get(cluster, "Unknown")

st.subheader("🔮 Prediction Result")
st.success(f"Cluster: {cluster} | Segment: {segment}")

# -------------------------
# PCA VISUALIZATION
# -------------------------
input_processed = preprocessor.transform(input_df)
input_pca = pca.transform(input_processed)

fig, ax = plt.subplots()

# Plot clusters
scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels
)

# Plot user point
ax.scatter(
    input_pca[0, 0],
    input_pca[0, 1],
    marker='*',
    s=300,
    label="Your Input"
)

ax.set_title("Customer Segments (PCA View)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()

st.pyplot(fig)

# -------------------------
# OPTIONAL INSIGHT
# -------------------------
st.subheader("📊 Dataset Overview")
st.write(df.head())