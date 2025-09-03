class Executor:
    def execute(self, action: dict) -> str:
        if action["action"] == "fetch_weather":
            return "Сегодня солнечно, +25°C"
        elif action["action"] == "fetch_news":
            return "Главная новость: AI изменяет мир!"
        else:
            return "Извини, я пока не понял твой запрос."
