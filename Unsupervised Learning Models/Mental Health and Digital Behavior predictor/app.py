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
MODEL_PATH = "model/model.pkl"
DATA_PATH = "data/mental_health_digital_behavior_data.csv"

SEGMENT_MAP = {
    0: "Healthy Balanced Users",
    1: "Digitally Overloaded Users",
    2: "At-Risk Users"
}

SEGMENT_DESCRIPTION = {
    0: "Good sleep, low anxiety, and high wellbeing. Healthy digital habits.",
    1: "High screen time, low sleep, and very high anxiety. Needs attention.",
    2: "Moderate usage but elevated anxiety. Potential hidden risk group."
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

X = df.copy()

# -------------------------
# PREPROCESSING FROM PIPELINE
# -------------------------
# model[:-1] = preprocessing part (scaler)
X_processed = model[:-1].transform(X)

# PCA for visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

# cluster labels
labels = model.predict(X)

# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Mental Health Segmentation", layout="wide")

st.title("🧠 Mental Health & Digital Behavior Segmentation")
st.write("Analyze how digital habits impact mental wellbeing")

# -------------------------
# SIDEBAR INPUT
# -------------------------
st.sidebar.header("📥 User Input")

screen_time = st.sidebar.slider("Daily Screen Time (min)", 100, 600, 300)
app_switches = st.sidebar.slider("App Switches", 10, 100, 50)
sleep = st.sidebar.slider("Sleep Hours", 3.0, 10.0, 7.0)
notifications = st.sidebar.slider("Notifications", 10, 150, 60)
social_media = st.sidebar.slider("Social Media Time (min)", 0, 300, 100)
focus = st.sidebar.slider("Focus Score", 1.0, 10.0, 6.0)
mood = st.sidebar.slider("Mood Score", 1.0, 10.0, 7.0)
anxiety = st.sidebar.slider("Anxiety Level", 1.0, 10.0, 5.0)
wellbeing = st.sidebar.slider("Wellbeing Score", 1.0, 100.0, 50.0)

# -------------------------
# PREDICTION
# -------------------------
input_df = pd.DataFrame([{
    "daily_screen_time_min": screen_time,
    "num_app_switches": app_switches,
    "sleep_hours": sleep,
    "notification_count": notifications,
    "social_media_time_min": social_media,
    "focus_score": focus,
    "mood_score": mood,
    "anxiety_level": anxiety,
    "digital_wellbeing_score": wellbeing
}])

cluster = model.predict(input_df)[0]
segment = SEGMENT_MAP.get(cluster, "Unknown")
description = SEGMENT_DESCRIPTION.get(cluster, "")

# -------------------------
# OUTPUT
# -------------------------
st.subheader("🔮 Prediction Result")

col1, col2 = st.columns(2)

with col1:
    st.metric("User Type", segment)

with col2:
    st.metric("Cluster ID", cluster)

st.info(f"🧾 Insight: {description}")

# -------------------------
# PCA VISUALIZATION
# -------------------------
input_processed = model[:-1].transform(input_df)
input_pca = pca.transform(input_processed)

fig, ax = plt.subplots()

scatter = ax.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    c=labels
)

# highlight user
ax.scatter(
    input_pca[0, 0],
    input_pca[0, 1],
    marker='*',
    s=300,
    label="You"
)

ax.set_title("User Segments (PCA View)")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.legend()

st.pyplot(fig)

# -------------------------
# OPTIONAL DATA VIEW
# -------------------------
with st.expander("📊 View Dataset"):
    st.dataframe(df.head())