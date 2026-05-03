# app.py

import customtkinter as ctk
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline

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
# LOAD MODEL + DATA
# -------------------------
model: Pipeline = joblib.load(MODEL_PATH)
data = pd.read_csv(DATA_PATH)

X = data.values

# Get trained parts
scaler = model["scaler"]
kmeans = model["kmeans"]

X_scaled = scaler.transform(X)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_


# -------------------------
# PREDICTION FUNCTION
# -------------------------
def predict_and_plot():
    try:
        x_val = float(entry_x.get())
        y_val = float(entry_y.get())

        input_data = np.array([[x_val, y_val]])
        input_scaled = scaler.transform(input_data)

        cluster = kmeans.predict(input_scaled)[0]
        segment = SEGMENT_MAP.get(cluster, "Unknown")

        result_label.configure(
            text=f"Cluster: {cluster} | Segment: {segment}"
        )

        # -------------------------
        # PLOT
        # -------------------------
        plt.figure()

        # Existing clusters
        plt.scatter(X_scaled[labels == 0, 0], X_scaled[labels == 0, 1])
        plt.scatter(X_scaled[labels == 1, 0], X_scaled[labels == 1, 1])
        plt.scatter(X_scaled[labels == 2, 0], X_scaled[labels == 2, 1])

        # Centroids
        plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200)

        # New point (highlight)
        plt.scatter(input_scaled[0, 0], input_scaled[0, 1], marker='*', s=300)

        plt.title("KMeans Clustering Visualization")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")

        plt.show()

    except Exception as e:
        result_label.configure(text=f"Error: {str(e)}")


# -------------------------
# UI SETUP
# -------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("KMeans Clustering 2D App")
app.geometry("400x300")

title = ctk.CTkLabel(app, text="KMeans Cluster Predictor", font=("Arial", 20))
title.pack(pady=10)

entry_x = ctk.CTkEntry(app, placeholder_text="Enter X value")
entry_x.pack(pady=10)

entry_y = ctk.CTkEntry(app, placeholder_text="Enter Y value")
entry_y.pack(pady=10)

predict_btn = ctk.CTkButton(app, text="Predict & Show Graph", command=predict_and_plot)
predict_btn.pack(pady=15)

result_label = ctk.CTkLabel(app, text="")
result_label.pack(pady=10)

app.mainloop()