import streamlit as st
import requests

API_CHAT_URL = "http://127.0.0.1:8000/api/chat"

def main():
    st.set_page_config(page_title="TezAI Chat", layout="centered")

    st.title("💬 TezAI Chat")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    user_input = st.text_input("Введите сообщение:")

    if st.button("Отправить") and user_input.strip():
        st.session_state.messages.append(("user", user_input))

        try:
            response = requests.post(API_CHAT_URL, json={"message": user_input})
            bot_reply = response.json().get("reply", "Ошибка в ответе")
        except Exception as e:
            bot_reply = f"Ошибка подключения: {e}"

        st.session_state.messages.append(("bot", bot_reply))

    # Отображаем чат
    for sender, text in st.session_state.messages:
        if sender == "user":
            st.markdown(f"**Вы:** {text}")
        else:
            st.markdown(f"**TezAI:** {text}")

if __name__ == "__main__":
    main()