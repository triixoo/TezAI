class Planner:
    def plan(self, parsed: dict) -> dict:
        intent = parsed.get("intent")
        if intent == "get_weather":
            return {"action": "fetch_weather"}
        elif intent == "get_news":
            return {"action": "fetch_news"}
        else:
            return {"action": "fallback"}