from fastapi import APIRouter, WebSocket

router = APIRouter()

@router.websocket("/chat")
async def websocket_chat(websocket: WebSocket):
    await websocket.accept()
    await websocket.send_text("Добро пожаловать в WebSocket API!")

    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"Эхо: {data}")
    except Exception:
        await websocket.close()
