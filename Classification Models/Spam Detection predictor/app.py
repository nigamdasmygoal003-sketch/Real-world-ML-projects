# app.py

import customtkinter as ctk
from src.predict import predict

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")


class ChatApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Spam Message Detector")
        self.geometry("600x700")

        # Title
        title = ctk.CTkLabel(self, text="Spam Detector Chat", font=("Arial", 20, "bold"))
        title.pack(pady=10)

        # Chat area (scrollable)
        self.chat_frame = ctk.CTkScrollableFrame(self)
        self.chat_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Input frame
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.pack(fill="x", padx=10, pady=10)

        self.entry = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message...")
        self.entry.pack(side="left", fill="x", expand=True, padx=(5, 5), pady=5)

        send_btn = ctk.CTkButton(self.input_frame, text="Send", command=self.send_message)
        send_btn.pack(side="right", padx=5)

    # ---------- Chat Bubble ----------
    def add_message(self, text, sender="user"):
        bubble = ctk.CTkFrame(self.chat_frame, corner_radius=10)

        if sender == "user":
            bubble.configure(fg_color="#2563eb")  # blue
            anchor = "e"
            text_color = "white"
        else:
            bubble.configure(fg_color="#374151")  # gray
            anchor = "w"
            text_color = "white"

        label = ctk.CTkLabel(bubble, text=text, wraplength=350, justify="left", text_color=text_color)
        label.pack(padx=10, pady=5)

        bubble.pack(anchor=anchor, padx=10, pady=5)

    # ---------- Send Message ----------
    def send_message(self):
        message = self.entry.get().strip()

        if not message:
            return

        # Show user message
        self.add_message(message, sender="user")

        # Clear input
        self.entry.delete(0, "end")

        # Predict
        result = predict(message)

        if "error" in result:
            self.add_message(f"Error: {result['error']}", sender="bot")
            return

        prob = result["spam_probability"]

        if result["spam"]:
            reply = f"🚨 SPAM DETECTED\nProbability: {prob}"
        else:
            reply = f"✅ SAFE MESSAGE\nProbability: {prob}"

        # Show bot response
        self.add_message(reply, sender="bot")


# Run App
if __name__ == "__main__":
    app = ChatApp()
    app.mainloop()