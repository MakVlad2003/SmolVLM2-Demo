import os
import uuid
import queue
import logging
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from . import config

logger = logging.getLogger(__name__)


class ApiHandler:
    """
    Обёртка над FastAPI-приложением для HTTP API.
    Эндпоинты:
      - POST /ptt/convert  — VQA / captioning (chat-режим)
      - POST /ptt/ocr      — OCR-распознавание текста
    """

    def __init__(
        self,
        task_queue: "queue.Queue[Dict[str, Any]]",
        result_broker,
        storage_dir: str = "uploads",
    ) -> None:
        self.task_queue = task_queue
        self.result_broker = result_broker
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        app = FastAPI(title="VLM API", version="1.0.0")
        self.app = app

        @app.post("/convert")
        async def convert(
            image: Optional[UploadFile] = File(default=None),
            query: str = Form(..., description="User question / prompt"),
        ):
            """
            Визуальный вопрос-ответ + captioning:
              - query (обязательный)
              - image (опционально, если не передан — используем DEMO_IMAGE)
            """
            if not query or not query.strip():
                raise HTTPException(status_code=400, detail="Поле 'query' обязательно и не может быть пустым.")

            # Если изображение не передано — берём демо-картинку
            if image is None:
                demo_path = Path(config.DEMO_IMAGE)
                if not demo_path.exists():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Демо-изображение не найдено по пути {demo_path}",
                    )
                image_path = demo_path
            else:
                # Проверим, что это изображение (content-type начинается с 'image/')
                content_type = (image.content_type or "").lower()
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Ожидался файл изображения, получен тип '{content_type}'.",
                    )

                # Сохраним на диск
                suffix = Path(image.filename or "").suffix or ".png"
                fname = f"{uuid.uuid4().hex}{suffix}"
                image_path = self.storage_dir / fname

                try:
                    raw = await image.read()
                    image_path.write_bytes(raw)
                except Exception as e:
                    logger.exception("Не удалось сохранить загруженный файл")
                    raise HTTPException(
                        status_code=500,
                        detail=f"Не удалось сохранить файл: {e}",
                    )

            # Создаём задачу для воркера
            task_id = uuid.uuid4().int & ((1 << 31) - 1)
            self.task_queue.put(
                {
                    "id": task_id,
                    "image_path": str(image_path),
                    "prompt": query,
                    "mode": "chat",  # режим VQA/captioning
                }
            )

            waiter = self.result_broker.register(task_id)

            def wait_for_result(timeout: int = config.INFERENCE_TIMEOUT):
                try:
                    return waiter.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError("Время ожидания ответа модели истекло.")

            try:
                result = await run_in_threadpool(wait_for_result)
            except TimeoutError as e:
                raise HTTPException(status_code=504, detail=str(e))

            if "error" in result:
                return JSONResponse(
                    status_code=500,
                    content={"id": task_id, "error": result["error"]},
                )

            return {"id": task_id, "result": result.get("result", "")}

        @app.post("/ocr")
        async def ocr(
            image: UploadFile = File(..., description="Изображение с текстом"),
        ):
            """
            OCR сценaрий:
              - обязательно изображение
              - внутренний OCR-промпт задаётся в config.OCR_SYSTEM_PROMPT
            """
            content_type = (image.content_type or "").lower()
            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"Для OCR ожидается изображение, получен тип '{content_type}'.",
                )

            suffix = Path(image.filename or "").suffix or ".png"
            fname = f"{uuid.uuid4().hex}{suffix}"
            image_path = self.storage_dir / fname

            try:
                raw = await image.read()
                image_path.write_bytes(raw)
            except Exception as e:
                logger.exception("Не удалось сохранить файл для OCR")
                raise HTTPException(
                    status_code=500,
                    detail=f"Не удалось сохранить файл: {e}",
                )

            task_id = uuid.uuid4().int & ((1 << 31) - 1)
            self.task_queue.put(
                {
                    "id": task_id,
                    "image_path": str(image_path),
                    "prompt": "",  # для OCR текст промпта не нужен
                    "mode": "ocr",
                }
            )

            waiter = self.result_broker.register(task_id)

            def wait_for_result(timeout: int = config.INFERENCE_TIMEOUT):
                try:
                    return waiter.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError("Время ожидания OCR-результата истекло.")

            try:
                result = await run_in_threadpool(wait_for_result)
            except TimeoutError as e:
                raise HTTPException(status_code=504, detail=str(e))

            if "error" in result:
                return JSONResponse(
                    status_code=500,
                    content={"id": task_id, "error": result["error"]},
                )

            return {"id": task_id, "result": result.get("result", "")}
