# app.py

import customtkinter as ctk
import json
from src.predict import predict

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class FraudApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Credit Card Fraud Detector")
        self.geometry("700x650")

        # Title
        title = ctk.CTkLabel(self, text="Fraud Detection System", font=("Arial", 24, "bold"))
        title.pack(pady=10)

        # Tabs
        self.tabview = ctk.CTkTabview(self)
        self.tabview.pack(fill="both", expand=True, padx=20, pady=10)

        self.tab_manual = self.tabview.add("Quick Input")
        self.tab_json = self.tabview.add("JSON Input")

        # ---------------- QUICK INPUT ----------------
        self.amount = self.create_entry(self.tab_manual, "Amount")

        self.v1 = self.create_entry(self.tab_manual, "V1")
        self.v2 = self.create_entry(self.tab_manual, "V2")
        self.v3 = self.create_entry(self.tab_manual, "V3")
        self.v4 = self.create_entry(self.tab_manual, "V4")

        btn1 = ctk.CTkButton(
            self.tab_manual,
            text="Predict",
            command=self.predict_manual
        )
        btn1.pack(pady=10)

        # ---------------- JSON INPUT ----------------
        self.textbox = ctk.CTkTextbox(self.tab_json, height=250)
        self.textbox.pack(padx=10, pady=10, fill="both")

        # sample JSON
        self.textbox.insert("0.0", """{
    "V1": -1.359807,
    "V2": -0.072781,
    "V3": 2.536347,
    "V4": 1.378155,
    "V5": -0.338321,
    "V6": 0.462388,
    "V7": 0.239599,
    "V8": 0.098698,
    "V9": 0.363787,
    "V10": 0.090794,
    "V11": -0.551600,
    "V12": -0.617801,
    "V13": -0.991390,
    "V14": -0.311169,
    "V15": 1.468177,
    "V16": -0.470401,
    "V17": 0.207971,
    "V18": 0.025791,
    "V19": 0.403993,
    "V20": 0.251412,
    "V21": -0.018307,
    "V22": 0.277838,
    "V23": -0.110474,
    "V24": 0.066928,
    "V25": 0.128539,
    "V26": -0.189115,
    "V27": 0.133558,
    "V28": -0.021053,
    "Amount": 149.62
}""")

        btn2 = ctk.CTkButton(
            self.tab_json,
            text="Predict from JSON",
            command=self.predict_json
        )
        btn2.pack(pady=10)

        # ---------------- RESULT ----------------
        self.result_label = ctk.CTkLabel(self, text="", font=("Arial", 18))
        self.result_label.pack(pady=10)

    # ---------- UI helper ----------
    def create_entry(self, parent, label):
        lbl = ctk.CTkLabel(parent, text=label)
        lbl.pack(anchor="w", padx=10)

        entry = ctk.CTkEntry(parent)
        entry.pack(padx=10, pady=5, fill="x")

        return entry

    # ---------- Manual prediction ----------
    def predict_manual(self):
        try:
            # minimal input → fill rest as 0
            data = {
                "Amount": float(self.amount.get()),
                "V1": float(self.v1.get()),
                "V2": float(self.v2.get()),
                "V3": float(self.v3.get()),
                "V4": float(self.v4.get())
            }

            # fill missing features
            for i in range(1, 29):
                key = f"V{i}"
                if key not in data:
                    data[key] = 0.0

            result = predict(data)

            self.show_result(result)

        except ValueError:
            self.result_label.configure(text="Invalid input!", text_color="red")

    # ---------- JSON prediction ----------
    def predict_json(self):
        try:
            text = self.textbox.get("0.0", "end").strip()
            data = json.loads(text)

            result = predict(data)
            self.show_result(result)

        except Exception as e:
            self.result_label.configure(text=str(e), text_color="red")

    # ---------- Result display ----------
    def show_result(self, result):
        if "error" in result:
            self.result_label.configure(text=result["error"], text_color="red")
            return

        fraud = result["fraud"]
        prob = result["fraud_probability"]

        text = "🚨 FRAUD DETECTED" if fraud else "✅ NORMAL TRANSACTION"

        self.result_label.configure(
            text=f"{text}\nProbability: {prob}",
            text_color="red" if fraud else "green"
        )


if __name__ == "__main__":
    app = FraudApp()
    app.mainloop()