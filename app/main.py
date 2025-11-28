import logging
import queue
from typing import Any, Dict

from fastapi import FastAPI
from fastapi.responses import PlainTextResponse, RedirectResponse
from gradio.routes import mount_gradio_app

from . import config
from .inference import InferenceWorker
from .result_broker import ResultBroker
from .api_handler import ApiHandler
from .ui import GradioUI


def create_app() -> FastAPI:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )

    # Общая очередь задач для модели
    task_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue()

    # Брокер результатов
    broker = ResultBroker()

    # Воркер с моделью
    worker = InferenceWorker(
        task_queue=task_queue,
        result_queue=broker.incoming,
        model_id=config.MODEL_ID,
    )
    worker.start(warmup=True)

    # Строим Gradio UI
    ui_builder = GradioUI(task_queue=task_queue, result_broker=broker)
    demo = ui_builder.build()

    # FastAPI-приложение
    app = FastAPI(title="SmolVLM2 Demo — UI + API")

    @app.get("/", response_class=RedirectResponse)
    async def root_redirect():
        return "/ui"

    @app.get("/health", response_class=PlainTextResponse)
    async def health():
        return "ok"

    # Монтируем Gradio интерфейс на /ui
    mount_gradio_app(app, demo, path="/ui")

    # Монтируем API на /ptt
    api = ApiHandler(task_queue=task_queue, result_broker=broker)
    app.mount("/ptt", api.app)

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=config.PORT,
        reload=False,
    )
