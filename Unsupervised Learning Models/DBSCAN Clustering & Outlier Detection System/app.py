# app.py

import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# CONFIG
# -------------------------
MODEL_PATH = "model/dbscan_pipeline.pkl"
DATA_PATH = "data/DBSCAN.csv"

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

X = df[["Weight", "Height"]]

scaler = model["scaler"]
dbscan = model["dbscan"]

# scale training data
X_scaled = scaler.transform(X)

# existing labels
labels = dbscan.labels_

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="DBSCAN Outlier Detection", layout="wide")

st.title("📊 DBSCAN Clustering & Outlier Detection")
st.write("Detect whether a new data point is normal or an anomaly")

# -------------------------
# INPUT
# -------------------------
st.sidebar.header("Input")

weight = st.sidebar.slider("Weight", float(X["Weight"].min()), float(X["Weight"].max()), 65.0)
height = st.sidebar.slider("Height", float(X["Height"].min()), float(X["Height"].max()), 170.0)

input_point = np.array([[weight, height]])
input_scaled = scaler.transform(input_point)

# -------------------------
# OUTLIER DETECTION LOGIC
# -------------------------
# distance-based check
from sklearn.metrics.pairwise import euclidean_distances

distances = euclidean_distances(input_scaled, X_scaled)

# find nearest points
min_dist = distances.min()

# threshold based on eps
eps = dbscan.eps

if min_dist <= eps:
    result = "Normal (Inside Cluster)"
else:
    result = "⚠️ Outlier Detected"

# -------------------------
# OUTPUT
# -------------------------
st.subheader("🔮 Result")

if "Outlier" in result:
    st.error(result)
else:
    st.success(result)

st.write(f"Nearest Distance: {min_dist:.3f} | eps: {eps}")

# -------------------------
# VISUALIZATION
# -------------------------
fig, ax = plt.subplots()

# plot clusters
scatter = ax.scatter(
    X["Weight"],
    X["Height"],
    c=labels
)

# plot input point
ax.scatter(
    weight,
    height,
    color="red",
    s=200,
    marker="*",
    label="Your Point"
)

ax.set_title("DBSCAN Clusters")
ax.set_xlabel("Weight")
ax.set_ylabel("Height")
ax.legend()

st.pyplot(fig)

# -------------------------
# DATA VIEW
# -------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df.head())