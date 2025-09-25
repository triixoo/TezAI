class NLU:
    def parse(self, text: str) -> dict:
        if "погода" in text.lower():
            return {"intent": "get_weather", "entities": {}}
        elif "новости" in text.lower():
            return {"intent": "get_news", "entities": {}}
        else:
            return {"intent": "unknown", "entities": {}}
