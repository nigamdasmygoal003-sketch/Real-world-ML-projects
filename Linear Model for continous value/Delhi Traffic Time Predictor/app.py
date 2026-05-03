import customtkinter as ctk
import numpy as np
import joblib
from tkinter import messagebox

# Load models
scaler = joblib.load("scaler.pkl")
encoder = joblib.load("encoder.pkl")
model = joblib.load("model.pkl")

# App setup
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Delhi Traffic Time Predictor 🚗")
app.geometry("600x700")

# ===== SCROLLABLE FRAME =====
frame = ctk.CTkScrollableFrame(app, width=580, height=650)
frame.pack(pady=10, padx=10, fill="both", expand=True)

# ===== OPTIONS =====

areas = [
    "Vasant Kunj", "Kalkaji", "Greater Kailash", "Janakpuri",
    "Model Town", "Punjabi Bagh", "Dwarka", "Rohini", "Chandni Chowk"
]

time_of_day_options = ["Morning Peak", "Afternoon", "Evening Peak", "Night"]
day_options = ["Weekday", "Weekend"]
weather_options = ["Clear", "Rain", "Fog"]
traffic_options = ["Low", "Medium", "High"]
road_options = ["Highway", "Main Road", "Inner Road"]

# ===== UI HELPERS =====

def label(text):
    return ctk.CTkLabel(frame, text=text, font=("Arial", 14))

def entry():
    return ctk.CTkEntry(frame, width=250)

def dropdown(values):
    return ctk.CTkOptionMenu(frame, values=values, width=250)

# ===== INPUT FIELDS =====

label("Start Area").pack(pady=5)
start_area = dropdown(areas)
start_area.pack()

label("End Area").pack(pady=5)
end_area = dropdown(areas)
end_area.pack()

label("Distance (km)").pack(pady=5)
distance = entry()
distance.pack()

label("Average Speed (km/h)").pack(pady=5)
speed = entry()
speed.pack()

label("Time of Day").pack(pady=5)
time_menu = dropdown(time_of_day_options)
time_menu.pack()

label("Day Type").pack(pady=5)
day_menu = dropdown(day_options)
day_menu.pack()

label("Weather").pack(pady=5)
weather_menu = dropdown(weather_options)
weather_menu.pack()

label("Traffic Density").pack(pady=5)
traffic_menu = dropdown(traffic_options)
traffic_menu.pack()

label("Road Type").pack(pady=5)
road_menu = dropdown(road_options)
road_menu.pack()

# ===== PREDICT FUNCTION =====

def predict():
    try:
        num_data = np.array([[ 
            float(distance.get()),
            float(speed.get())
        ]])

        cat_data = np.array([[ 
            start_area.get(),
            end_area.get(),
            time_menu.get(),
            day_menu.get(),
            weather_menu.get(),
            traffic_menu.get(),
            road_menu.get()
        ]])

        num_scaled = scaler.transform(num_data)
        cat_encoded = encoder.transform(cat_data)

        final_input = np.hstack([num_scaled, cat_encoded])

        prediction = model.predict(final_input)[0]

        result_label.configure(
            text=f"Estimated Travel Time: {prediction:.2f} minutes",
            text_color="green"
        )

    except Exception as e:
        messagebox.showerror("Error", str(e))

# ===== BUTTON =====

predict_btn = ctk.CTkButton(
    frame,
    text="Predict Travel Time",
    command=predict,
    height=40,
    font=("Arial", 16, "bold")
)
predict_btn.pack(pady=20)

# ===== RESULT LABEL (BOLD + GREEN) =====

result_label = ctk.CTkLabel(
    frame,
    text="Result will appear here",
    font=("Arial", 18, "bold"),
    text_color="green"
)
result_label.pack(pady=20)

# Run app
app.mainloop()