import argparse
import uvicorn
import torch

from fastapi import FastAPI
from api.rest import router as rest_router
from api.websocket import router as ws_router

# UI
from ui import chat, dashboard

# Обучение
import training.train as train_script

# HuggingFace токенизатор и наша модель
from transformers import AutoTokenizer
from models.tezai.tezai_lm import TezAILM



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
    train_script.train()


def run_inference():
    """Интерактивный тест инференса"""
    tokenizer = AutoTokenizer.from_pretrained("./models/tezai_lm")
    model = TezAILM(tokenizer.vocab_size)
    model.load_state_dict(torch.load("./models/tezai_lm/pytorch_model.bin", map_location="cpu"))
    model.eval()

    while True:
        text = input("[User]: ")
        if text.lower() in ["exit", "quit"]:
            break
        input_ids = torch.tensor([tokenizer.encode(text)])
        output_text = model.generate(input_ids, tokenizer, max_new_tokens=30)
        print(f"[TezAI]: {output_text}")


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
