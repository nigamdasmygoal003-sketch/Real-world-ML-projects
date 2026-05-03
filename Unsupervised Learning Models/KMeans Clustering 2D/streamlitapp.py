# app.py

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model/kmeans_model.pkl"
DATA_PATH = "data/dataset.csv"

SEGMENT_MAP = {
    0: "Cluster 1",
    1: "Cluster 2",
    2: "Cluster 3"
}

# -------------------------
# LOAD MODEL
# -------------------------
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

model = load_model()
data = load_data()

scaler = model["scaler"]
kmeans = model["kmeans"]

X = data.values
X_scaled = scaler.transform(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="KMeans Clustering App", layout="centered")

st.title("📊 KMeans Clustering 2D App")
st.write("Enter values to see which cluster the point belongs to")

# Sidebar input
st.sidebar.header("Input Features")

x_val = st.sidebar.slider("X Value", float(data["x"].min()), float(data["x"].max()), 0.0)
y_val = st.sidebar.slider("Y Value", float(data["y"].min()), float(data["y"].max()), 0.0)

# -------------------------
# PREDICTION
# -------------------------
input_data = np.array([[x_val, y_val]])
input_scaled = scaler.transform(input_data)

cluster = kmeans.predict(input_scaled)[0]
segment = SEGMENT_MAP.get(cluster, "Unknown")

st.subheader("🔮 Prediction")
st.success(f"Cluster: {cluster} | Segment: {segment}")

# -------------------------
# PLOT
# -------------------------
fig, ax = plt.subplots()

# clusters
ax.scatter(X_scaled[labels == 0, 0], X_scaled[labels == 0, 1], label="Cluster 0")
ax.scatter(X_scaled[labels == 1, 0], X_scaled[labels == 1, 1], label="Cluster 1")
ax.scatter(X_scaled[labels == 2, 0], X_scaled[labels == 2, 1], label="Cluster 2")

# centroids
ax.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, label="Centroids")

# user point
ax.scatter(input_scaled[0, 0], input_scaled[0, 1], marker='*', s=300, label="Your Point")

ax.set_title("Cluster Visualization (Scaled Space)")
ax.legend()

st.pyplot(fig)