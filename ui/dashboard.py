import streamlit as st
import requests

API_URL = "http://127.0.0.1:8000/api/status"

def main():
    st.set_page_config(page_title="TezAI Dashboard", layout="wide")

    st.title("📊 TezAI Dashboard")
    st.write("Веб-панель для управления ассистентом")

    if st.button("Проверить API"):
        try:
            response = requests.get(API_URL)
            st.success(response.json())
        except Exception as e:
            st.error(f"Ошибка подключения: {e}")

    st.subheader("⚙ Настройки обучения")
    dataset_path = st.text_input("Путь к датасету", "data/cleaned/")
    epochs = st.slider("Кол-во эпох", 1, 10, 3)

    if st.button("Запустить обучение"):
        st.info(f"Запуск обучения на {epochs} эпох с датасетом {dataset_path}... (заглушка)")

if __name__ == "__main__":
    main()