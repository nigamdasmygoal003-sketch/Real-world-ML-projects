import numpy as np
import customtkinter as ctk
import joblib 

# Load Model

try:
    model = joblib.load("model.pkl")
    scaler = joblib.load("scaler.pkl")

except:
    print("Error in loading model and scaler !")

# App setting

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

app = ctk.CTk()
app.title("Student Performance")
app.geometry("500x700")

# Title

Title = ctk.CTkLabel(
    app,
    text = "Student Performance",
    font = ("Arial",22,"bold")
)

Title.pack(pady =15)

# Scrollable fram

frame = ctk.CTkScrollableFrame(app,width=450,height=550)
frame.pack(pady=10,padx=20,fill="both",expand=True)

# Input Fields

labels = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced"
]

entries = []

for label in labels:
    ctk.CTkLabel(frame,text=label).pack(pady=(10,2))
    entry = ctk.CTkEntry(frame,placeholder_text=f"Enter {label}")
    entry.pack(pady=(0,10),fill="x")
    entries.append(entry)


# Prediction Function

def predict():
    try:
        values = [float(e.get()) for e in entries]

        features = np.array([values])
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        StudentPerformance = prediction[0]

        result_lable.configure(
            text = f"Estimated Student Performance:\n{StudentPerformance:,.2f}",
            text_color = "green"
        )
    except:
        result_lable.configure(
            text="⚠️ Please enter valid numbers!",
            text_color="red"
        )


# Button 
predict_bnt = ctk.CTkButton(
    frame,
    text="Predict Performance",
    command=predict,
    height=40
)            
predict_bnt.pack(pady=20)

# Result Lable (Inside Frame)

result_lable = ctk.CTkLabel(
    frame,
    text="",
    font=("Arial",20,"bold")
)
result_lable.pack(pady=25)

# Run App

app.mainloop()