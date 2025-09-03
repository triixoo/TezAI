from fastapi import APIRouter
from pydantic import BaseModel

from agent.nlu import NLU
from agent.planner import Planner
from agent.executor import Executor
from agent.response import Responder

router = APIRouter()

# Модель для входящего запроса
class ChatRequest(BaseModel):
    message: str

# /status проверка
@router.get("/status")
async def get_status():
    return {"status": "ok", "message": "REST API работает"}

# /chat основной чат-эндпоинт
@router.post("/chat")
async def chat_endpoint(req: ChatRequest):
    user_message = req.message

    # --- пайплайн NLU → Planner → Executor → Responder ---
    nlu = NLU()
    planner = Planner()
    executor = Executor()
    responder = Responder()

    parsed = nlu.parse(user_message)
    plan = planner.plan(parsed)
    result = executor.execute(plan)
    reply = responder.format(result)

    return {"reply": reply}
