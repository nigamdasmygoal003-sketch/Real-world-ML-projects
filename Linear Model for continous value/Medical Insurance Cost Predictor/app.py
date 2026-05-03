import customtkinter as ctk
import numpy as np
import joblib
from tkinter import messagebox

# Load saved models
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
model = joblib.load("model.pkl")

# App setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Medical Insurance Cost Predictor 💰")
app.geometry("500x600")

# ===== SCROLLABLE FRAME =====
frame = ctk.CTkScrollableFrame(app, width=480, height=550)
frame.pack(pady=10, padx=10, fill="both", expand=True)

# ===== OPTIONS =====
sex_options = ["male", "female"]
smoker_options = ["yes", "no"]
region_options = ["southwest", "southeast", "northwest", "northeast"]

# ===== UI HELPERS =====
def label(text):
    return ctk.CTkLabel(frame, text=text, font=("Arial", 14))

def entry():
    return ctk.CTkEntry(frame, width=250)

def dropdown(values):
    return ctk.CTkOptionMenu(frame, values=values, width=250)

# ===== INPUT FIELDS =====

label("Age").pack(pady=5)
age_entry = entry()
age_entry.pack()

label("BMI").pack(pady=5)
bmi_entry = entry()
bmi_entry.pack()

label("Children").pack(pady=5)
children_entry = entry()
children_entry.pack()

label("Sex").pack(pady=5)
sex_menu = dropdown(sex_options)
sex_menu.pack()

label("Smoker").pack(pady=5)
smoker_menu = dropdown(smoker_options)
smoker_menu.pack()

label("Region").pack(pady=5)
region_menu = dropdown(region_options)
region_menu.pack()

# ===== PREDICT FUNCTION =====

def predict():
    try:
        # Convert inputs
        age = float(age_entry.get())
        bmi = float(bmi_entry.get())
        children = float(children_entry.get())

        sex = 1 if sex_menu.get() == "male" else 0
        smoker = 1 if smoker_menu.get() == "yes" else 0
        region = region_menu.get()

        # Numerical + already encoded columns
        num_data = np.array([[age, bmi, children, sex, smoker]])

        # Categorical (only region)
        cat_data = np.array([[region]])

        # Transform
        num_scaled = scaler.transform(num_data)
        cat_encoded = encoder.transform(cat_data)

        final_input = np.hstack([num_scaled, cat_encoded])

        # Predict
        prediction = model.predict(final_input)[0]

        result_label.configure(
            text=f"Estimated Charges: ₹ {prediction:,.2f}",
            text_color="green"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ===== BUTTON =====
predict_btn = ctk.CTkButton(
    frame,
    text="Predict Insurance Cost",
    command=predict,
    font=("Arial", 16, "bold"),
    height=40
)
predict_btn.pack(pady=20)

# ===== RESULT LABEL =====
result_label = ctk.CTkLabel(
    frame,
    text="Result will appear here",
    font=("Arial", 18, "bold"),
    text_color="green"
)
result_label.pack(pady=20)

# Run app
app.mainloop()