# app.py

import customtkinter as ctk
from src.predict import predict

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class LoanApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Loan Approval Predictor")
        self.geometry("500x600")

        # Title
        self.title_label = ctk.CTkLabel(
            self, text="Loan Approval System", font=("Arial", 22, "bold")
        )
        self.title_label.pack(pady=20)

        # Input Frame
        self.frame = ctk.CTkFrame(self)
        self.frame.pack(padx=20, pady=10, fill="both", expand=True)

        # Inputs
        self.city = self.create_entry("City")
        self.income = self.create_entry("Income")
        self.credit = self.create_entry("Credit Score")
        self.loan = self.create_entry("Loan Amount")
        self.years = self.create_entry("Years Employed")
        self.points = self.create_entry("Points (0–1)")

        # Predict Button
        self.predict_btn = ctk.CTkButton(
            self,
            text="Predict",
            command=self.make_prediction,
            height=40
        )
        self.predict_btn.pack(pady=20)

        # Output Label
        self.result_label = ctk.CTkLabel(
            self, text="", font=("Arial", 16)
        )
        self.result_label.pack(pady=10)

    def create_entry(self, placeholder):
        entry = ctk.CTkEntry(self.frame, placeholder_text=placeholder, height=35)
        entry.pack(pady=8, padx=20, fill="x")
        return entry

    def make_prediction(self):
        try:
            data = {
                "city": self.city.get(),
                "income": float(self.income.get()),
                "credit_score": float(self.credit.get()),
                "loan_amount": float(self.loan.get()),
                "years_employed": float(self.years.get()),
                "points": float(self.points.get())
            }

            result = predict(data)

            if "error" in result:
                self.result_label.configure(
                    text=f"Error: {result['error']}", text_color="red"
                )
                return

            prediction = "APPROVED ✅" if result["prediction"] else "REJECTED ❌"
            prob = result["approval_probability"]

            self.result_label.configure(
                text=f"{prediction}\nConfidence: {prob}",
                text_color="green" if result["prediction"] else "red"
            )

        except ValueError:
            self.result_label.configure(
                text="Please enter valid numeric values!", text_color="red"
            )


if __name__ == "__main__":
    app = LoanApp()
    app.mainloop()