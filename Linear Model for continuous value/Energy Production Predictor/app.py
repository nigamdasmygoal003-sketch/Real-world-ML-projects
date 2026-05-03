import customtkinter as ctk
import numpy as np
import joblib
from tkinter import messagebox

# Load saved models
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
model = joblib.load("model.pkl")

# Initialize app
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Energy Production Predictor")
app.geometry("500x600")

# ====== INPUT FIELDS ======

def create_label(text):
    return ctk.CTkLabel(app, text=text)

def create_entry():
    return ctk.CTkEntry(app)

# Numeric inputs
create_label("Start Hour").pack(pady=5)
start_hour = create_entry()
start_hour.pack()

create_label("End Hour").pack(pady=5)
end_hour = create_entry()
end_hour.pack()

create_label("Day of Year").pack(pady=5)
day_of_year = create_entry()
day_of_year.pack()

# Dropdown options
sources = ["Solar", "Wind", "Hydro"]
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
months = ["January", "February", "March", "April", "May", "June",
          "July", "August", "September", "October", "November", "December"]
seasons = ["Winter", "Spring", "Summer", "Fall"]

def create_option(values):
    return ctk.CTkOptionMenu(app, values=values)

create_label("Source").pack(pady=5)
source_menu = create_option(sources)
source_menu.pack()

create_label("Day Name").pack(pady=5)
day_menu = create_option(days)
day_menu.pack()

create_label("Month Name").pack(pady=5)
month_menu = create_option(months)
month_menu.pack()

create_label("Season").pack(pady=5)
season_menu = create_option(seasons)
season_menu.pack()

# ====== PREDICTION FUNCTION ======

def predict():
    try:
        # Get values
        num_data = np.array([[ 
            int(start_hour.get()),
            int(end_hour.get()),
            int(day_of_year.get())
        ]])

        cat_data = np.array([[ 
            source_menu.get(),
            day_menu.get(),
            month_menu.get(),
            season_menu.get()
        ]])

        # Preprocess
        num_scaled = scaler.transform(num_data)
        cat_encoded = encoder.transform(cat_data)

        final_input = np.hstack([num_scaled, cat_encoded])

        # Predict
        prediction = model.predict(final_input)[0]

        result_label.configure(text=f"Predicted Production: {prediction:.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ====== BUTTON ======

predict_btn = ctk.CTkButton(app, text="Predict", command=predict)
predict_btn.pack(pady=20)

# ====== RESULT ======

result_label = ctk.CTkLabel(app, text="Prediction will appear here")
result_label.pack(pady=20)

# Run app
app.mainloop()