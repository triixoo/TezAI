import argparse
import uvicorn

from api.rest import router as rest_router
from api.websocket import router as ws_router
from fastapi import FastAPI

from ui import chat, dashboard
from training import run_train
from core.tokenizer import SimpleTokenizer
from core.model import SimpleModel
from core.inference import InferenceEngine


def run_api():
    """Запуск REST + WebSocket API"""
    app = FastAPI(title="TezAI API")
    app.include_router(rest_router, prefix="/api")
    app.include_router(ws_router, prefix="/api")
    uvicorn.run(app, host="0.0.0.0", port=8000)


def run_chat():
    """Запуск Streamlit-чата"""
    import os
    os.system("streamlit run ui/chat.py")


def run_dashboard():
    """Запуск Streamlit-дэшборда"""
    import os
    os.system("streamlit run ui/dashboard.py")


def run_training():
    """Запуск обучения модели"""
    run_train.train()


def run_inference():
    """Тестовый инференс"""
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(["погода сегодня хорошая", "новости важные"])
    model = SimpleModel(vocab_size=len(tokenizer.vocab))
    engine = InferenceEngine(model, tokenizer)

    text = "погода"
    print(f"[User]: {text}")
    print(f"[TezAI]: {engine.generate(text)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Запуск TezAI")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["api", "chat", "dashboard", "train", "inference"],
                        help="Режим запуска")
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
    elif args.mode == "chat":
        run_chat()
    elif args.mode == "dashboard":
        run_dashboard()
    elif args.mode == "train":
        run_training()
    elif args.mode == "inference":
        run_inference()
