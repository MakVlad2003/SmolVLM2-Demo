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
            if not query or not query.strip():
                raise HTTPException(status_code=400, detail="'query' is nessesary, it can't be empty.")

            if image is None:
                demo_path = Path(config.DEMO_IMAGE)
                if not demo_path.exists():
                    raise HTTPException(
                        status_code=500,
                        detail=f"Demo-image was not found by this path: {demo_path}",
                    )
                image_path = demo_path
            else:
                content_type = (image.content_type or "").lower()
                if not content_type.startswith("image/"):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Image file is wated, get a type: '{content_type}'.",
                    )

                suffix = Path(image.filename or "").suffix or ".png"
                fname = f"{uuid.uuid4().hex}{suffix}"
                image_path = self.storage_dir / fname

                try:
                    raw = await image.read()
                    image_path.write_bytes(raw)
                except Exception as e:
                    logger.exception("I can't find this file")
                    raise HTTPException(
                        status_code=500,
                        detail=f"I can't find this file: {e}",
                    )

            task_id = uuid.uuid4().int & ((1 << 31) - 1)
            self.task_queue.put(
                {
                    "id": task_id,
                    "image_path": str(image_path),
                    "prompt": query,
                    "mode": "chat",  
                }
            )

            waiter = self.result_broker.register(task_id)

            def wait_for_result(timeout: int = config.INFERENCE_TIMEOUT):
                try:
                    return waiter.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError("Time of model wating is finish.")

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
            content_type = (image.content_type or "").lower()
            if not content_type.startswith("image/"):
                raise HTTPException(
                    status_code=400,
                    detail=f"For OCR we wait image, type: '{content_type}'.",
                )

            suffix = Path(image.filename or "").suffix or ".png"
            fname = f"{uuid.uuid4().hex}{suffix}"
            image_path = self.storage_dir / fname

            try:
                raw = await image.read()
                image_path.write_bytes(raw)
            except Exception as e:
                logger.exception("I can't save the inage")
                raise HTTPException(
                    status_code=500,
                    detail=f"I can't save this file: {e}",
                )

            task_id = uuid.uuid4().int & ((1 << 31) - 1)
            self.task_queue.put(
                {
                    "id": task_id,
                    "image_path": str(image_path),
                    "prompt": "",  
                    "mode": "ocr",
                }
            )

            waiter = self.result_broker.register(task_id)

            def wait_for_result(timeout: int = config.INFERENCE_TIMEOUT):
                try:
                    return waiter.get(timeout=timeout)
                except queue.Empty:
                    raise TimeoutError("Time of wating OCR is finish.")

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
