# app.py

import customtkinter as ctk
from src.predict import predict

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChurnApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Customer Churn Predictor")
        self.geometry("700x700")

        title = ctk.CTkLabel(self, text="Customer Churn Prediction", font=("Arial", 24, "bold"))
        title.pack(pady=10)

        # ✅ Scrollable Frame
        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.pack(fill="both", expand=True, padx=20, pady=10)

        # Inputs
        self.gender = self.create_dropdown("Gender", ["Male", "Female"])
        self.senior = self.create_dropdown("Senior Citizen", ["0", "1"])
        self.partner = self.create_dropdown("Partner", ["Yes", "No"])
        self.dependents = self.create_dropdown("Dependents", ["Yes", "No"])

        self.tenure = self.create_entry("Tenure (months)")
        self.monthly = self.create_entry("Monthly Charges")
        self.total = self.create_entry("Total Charges")

        self.phone = self.create_dropdown("Phone Service", ["Yes", "No"])
        self.multiple = self.create_dropdown("Multiple Lines", ["Yes", "No", "No phone service"])

        self.internet = self.create_dropdown("Internet Service", ["DSL", "Fiber optic", "No"])
        self.security = self.create_dropdown("Online Security", ["Yes", "No", "No internet service"])
        self.backup = self.create_dropdown("Online Backup", ["Yes", "No", "No internet service"])
        self.device = self.create_dropdown("Device Protection", ["Yes", "No", "No internet service"])
        self.tech = self.create_dropdown("Tech Support", ["Yes", "No", "No internet service"])
        self.tv = self.create_dropdown("Streaming TV", ["Yes", "No", "No internet service"])
        self.movies = self.create_dropdown("Streaming Movies", ["Yes", "No", "No internet service"])

        self.contract = self.create_dropdown("Contract", ["Month-to-month", "One year", "Two year"])
        self.paperless = self.create_dropdown("Paperless Billing", ["Yes", "No"])
        self.payment = self.create_dropdown(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
        )

        # Predict Button
        btn = ctk.CTkButton(self, text="Predict", command=self.make_prediction, height=40)
        btn.pack(pady=10)

        # Result
        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 18))
        self.result_label.pack(pady=10)

    # ---------- UI Components ----------

    def create_entry(self, label_text):
        label = ctk.CTkLabel(self.scroll_frame, text=label_text)
        label.pack(anchor="w", padx=10)

        entry = ctk.CTkEntry(self.scroll_frame, height=35)
        entry.pack(pady=5, padx=10, fill="x")

        return entry

    def create_dropdown(self, label_text, values):
        label = ctk.CTkLabel(self.scroll_frame, text=label_text)
        label.pack(anchor="w", padx=10)

        dropdown = ctk.CTkOptionMenu(self.scroll_frame, values=values)
        dropdown.set(values[0])
        dropdown.pack(pady=5, padx=10, fill="x")

        return dropdown

    # ---------- Prediction ----------

    def make_prediction(self):
        try:
            data = {
                "gender": self.gender.get(),
                "SeniorCitizen": int(self.senior.get()),
                "Partner": self.partner.get(),
                "Dependents": self.dependents.get(),
                "tenure": float(self.tenure.get()),
                "PhoneService": self.phone.get(),
                "MultipleLines": self.multiple.get(),
                "InternetService": self.internet.get(),
                "OnlineSecurity": self.security.get(),
                "OnlineBackup": self.backup.get(),
                "DeviceProtection": self.device.get(),
                "TechSupport": self.tech.get(),
                "StreamingTV": self.tv.get(),
                "StreamingMovies": self.movies.get(),
                "Contract": self.contract.get(),
                "PaperlessBilling": self.paperless.get(),
                "PaymentMethod": self.payment.get(),
                "MonthlyCharges": float(self.monthly.get()),
                "TotalCharges": float(self.total.get())
            }

            result = predict(data)

            if "error" in result:
                self.result_label.configure(text=result["error"], text_color="red")
                return

            churn = "HIGH RISK ⚠️" if result["churn"] else "LOW RISK ✅"

            self.result_label.configure(
                text=f"{churn}\nProbability: {result['churn_probability']}",
                text_color="red" if result["churn"] else "green"
            )

        except ValueError:
            self.result_label.configure(text="Invalid input!", text_color="red")


if __name__ == "__main__":
    app = ChurnApp()
    app.mainloop()