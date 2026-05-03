import customtkinter as ctk
import numpy as np
import joblib
from tkinter import messagebox

# Load model
model = joblib.load("model.pkl")

# App setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Wine Quality Predictor 🍷")
app.geometry("600x750")

# ===== SCROLLABLE FRAME =====
frame = ctk.CTkScrollableFrame(app, width=580, height=700)
frame.pack(padx=10, pady=10, fill="both", expand=True)

# ===== INPUT FIELDS =====

def label(text):
    return ctk.CTkLabel(frame, text=text, font=("Arial", 14))

def entry():
    return ctk.CTkEntry(frame, width=250)

fields = {}

columns = [
    "fixed acidity", "volatile acidity", "citric acid",
    "residual sugar", "chlorides", "free sulfur dioxide",
    "total sulfur dioxide", "density", "pH",
    "sulphates", "alcohol"
]

for col in columns:
    label(col.title()).pack(pady=5)
    e = entry()
    e.pack()
    fields[col] = e

# ===== PREDICT FUNCTION =====

def predict():
    try:
        values = []

        for col in columns:
            val = fields[col].get()
            if val == "":
                raise ValueError(f"{col} cannot be empty")
            values.append(float(val))

        input_array = np.array([values])

        prediction = model.predict(input_array)[0]

        if prediction == 1:
            result_text = "🍷 Good Quality Wine"
            color = "green"
        else:
            result_text = "⚠️ Bad Quality Wine"
            color = "red"

        result_label.configure(
            text=result_text,
            text_color=color
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ===== BUTTON =====

predict_btn = ctk.CTkButton(
    frame,
    text="Predict Wine Quality",
    command=predict,
    font=("Arial", 16, "bold"),
    height=40
)
predict_btn.pack(pady=20)

# ===== RESULT =====

result_label = ctk.CTkLabel(
    frame,
    text="Result will appear here",
    font=("Arial", 20, "bold")
)
result_label.pack(pady=20)

# Run app
app.mainloop()