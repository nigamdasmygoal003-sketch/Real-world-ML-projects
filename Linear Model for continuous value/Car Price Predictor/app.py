import customtkinter as ctk
import numpy as np
import pandas as pd
import joblib

# =========================
# LOAD SAVED FILES
# =========================
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
num_imputer = joblib.load("num_imputer.pkl")
cat_imputer = joblib.load("cat_imputer.pkl")

# =========================
# APP SETTINGS
# =========================
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("🚗 Car Price Predictor")
app.geometry("600x750")

# =========================
# TITLE
# =========================
ctk.CTkLabel(
    app,
    text="🚗 Car Price Predictor",
    font=("Arial", 24, "bold")
).pack(pady=20)

# =========================
# FRAME
# =========================
frame = ctk.CTkScrollableFrame(app, width=550, height=600)
frame.pack(padx=20, pady=10, fill="both", expand=True)

# =========================
# INPUT FIELDS
# =========================
entries = {}

def add_entry(label):
    ctk.CTkLabel(frame, text=label).pack(pady=(10,2))
    entry = ctk.CTkEntry(frame)
    entry.pack(pady=(0,10), fill="x")
    entries[label] = entry

# Numerical inputs
add_entry("Levy")
add_entry("Prod. year")
add_entry("Engine volume")
add_entry("Mileage")
add_entry("Cylinders")
add_entry("Airbags")
add_entry("Doors")

# =========================
# DROPDOWNS (Categorical)
# =========================
def add_dropdown(label, values):
    var = ctk.StringVar(value=values[0])
    ctk.CTkLabel(frame, text=label).pack(pady=(10,2))
    menu = ctk.CTkOptionMenu(frame, values=values, variable=var)
    menu.pack(pady=5, fill="x")
    return var

manufacturer_var = add_dropdown("Manufacturer", ["Toyota","BMW","Mercedes","Ford","Honda"])
model_var = add_dropdown("Model", ["Corolla","X5","C-Class","Focus","Civic"])
category_var = add_dropdown("Category", ["Sedan","SUV","Hatchback"])
leather_var = add_dropdown("Leather interior", ["Yes","No"])
fuel_var = add_dropdown("Fuel type", ["Petrol","Diesel","Hybrid"])
gear_var = add_dropdown("Gear box type", ["Manual","Automatic"])
drive_var = add_dropdown("Drive wheels", ["FWD","RWD","4x4"])
wheel_var = add_dropdown("Wheel", ["Left wheel","Right-hand drive"])
color_var = add_dropdown("Color", ["Black","White","Silver","Blue","Red"])

# =========================
# PREDICT FUNCTION
# =========================
def predict():
    try:
        # Create input dictionary
        input_data = {
            "Levy": float(entries["Levy"].get()),
            "Manufacturer": manufacturer_var.get(),
            "Model": model_var.get(),
            "Prod. year": float(entries["Prod. year"].get()),
            "Category": category_var.get(),
            "Leather interior": leather_var.get(),
            "Fuel type": fuel_var.get(),
            "Engine volume": float(entries["Engine volume"].get()),
            "Mileage": float(entries["Mileage"].get()),
            "Cylinders": float(entries["Cylinders"].get()),
            "Gear box type": gear_var.get(),
            "Drive wheels": drive_var.get(),
            "Doors": float(entries["Doors"].get()),
            "Wheel": wheel_var.get(),
            "Color": color_var.get(),
            "Airbags": float(entries["Airbags"].get())
        }

        input_df = pd.DataFrame([input_data])

        # Separate columns
        num_cols = input_df.select_dtypes(include=["int64","float64"]).columns
        cat_cols = input_df.select_dtypes(include=["object"]).columns

        # Preprocess
        X_num = num_imputer.transform(input_df[num_cols])
        X_num = scaler.transform(X_num)

        X_cat = cat_imputer.transform(input_df[cat_cols])
        X_cat = encoder.transform(X_cat)

        X_final = np.hstack([X_num, X_cat])

        # Predict
        prediction = model.predict(X_final)[0]

        result_label.configure(
            text=f"💰 Estimated Price: ${prediction:,.2f}",
            text_color="green"
        )

    except Exception as e:
        result_label.configure(
            text=f"⚠️ Error: {str(e)}",
            text_color="red"
        )

# =========================
# BUTTON
# =========================
ctk.CTkButton(
    frame,
    text="Predict Price",
    command=predict,
    height=40
).pack(pady=20)

# =========================
# RESULT
# =========================
result_label = ctk.CTkLabel(
    frame,
    text="",
    font=("Arial", 20, "bold")
)
result_label.pack(pady=20)

# =========================
# RUN
# =========================
app.mainloop()