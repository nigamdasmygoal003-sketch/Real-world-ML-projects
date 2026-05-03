import streamlit as st
from src.predict import predict

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Spam Detector Chat", layout="centered")

st.title("💬 Spam Message Detector")
st.markdown("Chat-style spam detection using NLP")

# =========================
# SESSION STATE (CHAT HISTORY)
# =========================
if "messages" not in st.session_state:
    st.session_state.messages = []

# =========================
# DISPLAY CHAT HISTORY
# =========================
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# =========================
# USER INPUT
# =========================
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    # Predict
    result = predict(user_input)

    if "error" in result:
        bot_reply = f"⚠️ Error: {result['error']}"
    else:
        prob = result["spam_probability"]

        if result["spam"]:
            bot_reply = f"🚨 **SPAM DETECTED**\n\nProbability: {prob:.2f}"
        else:
            bot_reply = f"✅ **SAFE MESSAGE**\n\nProbability: {prob:.2f}"

    # Show bot message
    st.session_state.messages.append({"role": "assistant", "content": bot_reply})

    with st.chat_message("assistant"):
        st.markdown(bot_reply)