TezAI/
 ├── README.md
 ├── docker-compose.yml
 ├── Dockerfile
 ├── requirements.txt
 ├── data/                  # корпус данных (сырые и очищенные тексты)
 │    ├── raw/
 │    └── cleaned/
 ├── training/              # тренировка модели
 │    ├── tokenizer.py
 │    ├── preprocess.py
 │    ├── train.py          # скрипт запуска обучения (DeepSpeed/Megatron)
 │    └── configs/
 │         ├── tezai_1.3B.json
 │         └── tezai_3B.json
 ├── inference/             # инференс-сервер
 │    ├── server.py         # FastAPI + WebSocket
 │    ├── model_loader.py
 │    └── memory/           # модуль памяти (FAISS + JSON history)
 ├── agents/                # твои кастомные агенты
 │    ├── cli_agent.py
 │    ├── telegram_agent.py
 │    └── research_agent.py
 └── infra/                 # развертывание
      ├── k8s/
      └── monitoring/
