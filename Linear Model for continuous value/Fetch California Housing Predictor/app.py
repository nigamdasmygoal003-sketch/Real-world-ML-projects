import customtkinter as ctk
import numpy as np
import joblib

# =========================
# LOAD MODEL
# =========================
try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")
except:
    print("Error loading model or scaler!")

# =========================
# APP SETTINGS
# =========================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("🏠 California House Price Predictor")
app.geometry("500x700")

# =========================
# TITLE
# =========================
title = ctk.CTkLabel(
    app,
    text="🏠 California House Price Predictor",
    font=("Arial", 22, "bold")
)
title.pack(pady=15)

# =========================
# SCROLLABLE FRAME
# =========================
frame = ctk.CTkScrollableFrame(app, width=450, height=550)
frame.pack(pady=10, padx=20, fill="both", expand=True)

# =========================
# INPUT FIELDS
# =========================
labels = [
    "Median Income",
    "House Age",
    "Average Rooms",
    "Average Bedrooms",
    "Population",
    "Average Occupancy",
    "Latitude",
    "Longitude"
]

entries = []

for label in labels:
    ctk.CTkLabel(frame, text=label).pack(pady=(10, 2))
    entry = ctk.CTkEntry(frame, placeholder_text=f"Enter {label}")
    entry.pack(pady=(0, 10), fill="x")
    entries.append(entry)

# =========================
# PREDICT FUNCTION
# =========================
def predict():
    try:
        values = [float(e.get()) for e in entries]

        features = np.array([values])
        features_scaled = scaler.transform(features)

        prediction = model.predict(features_scaled)
        price = prediction[0] * 1000

        result_label.configure(
            text=f"💰 Estimated Price:\n${price:,.2f}",
            text_color="green"
        )

    except:
        result_label.configure(
            text="⚠️ Please enter valid numbers!",
            text_color="red"
        )

# =========================
# BUTTON
# =========================
predict_btn = ctk.CTkButton(
    frame,
    text="Predict Price",
    command=predict,
    height=40
)
predict_btn.pack(pady=20)

# =========================
# RESULT LABEL (INSIDE FRAME ✅)
# =========================
result_label = ctk.CTkLabel(
    frame,
    text="",
    font=("Arial", 20, "bold")
)
result_label.pack(pady=25)

# =========================
# RUN APP
# =========================
app.mainloop()