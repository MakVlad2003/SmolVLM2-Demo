import os
import queue
import threading
import logging
from pathlib import Path
from typing import Dict, Any

import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

from . import config

logger = logging.getLogger(__name__)


class InferenceWorker:
    """
    Отдельный поток, который:
    - грузит модель SmolVLM2;
    - принимает задачи из task_queue;
    - пишет результаты в result_queue (это broker.incoming).
    """

    def __init__(
        self,
        task_queue: "queue.Queue[Dict[str, Any]]",
        result_queue: "queue.Queue[Dict[str, Any]]",
        model_id: str | None = None,
    ) -> None:
        self.task_queue = task_queue
        self.result_queue = result_queue
        self.model_id = model_id or config.MODEL_ID

        self.device = self._resolve_device(config.DEVICE_MODE)
        self.dtype = torch.bfloat16 if self.device.type == "cuda" else torch.float32

        logger.info(f"[SmolVLM] Loading model {self.model_id} on device={self.device} ...")

        # Если выставить HF_LOCAL_ONLY=1, то from_pretrained будет работать только с локальным кэшем
        local_files_only = os.getenv("HF_LOCAL_ONLY", "0") == "1"

        # Убедимся, что директория кэша существует (важно при монтировании volume)
        try:
            config.MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning(f"[SmolVLM] Failed to create MODEL_CACHE_DIR {config.MODEL_CACHE_DIR}: {e}")

        self.processor = AutoProcessor.from_pretrained(
            self.model_id,
            local_files_only=local_files_only,
        )
        self.model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            _attn_implementation="sdpa",
            local_files_only=local_files_only,
        ).to(self.device)

        logger.info("[SmolVLM] Model loaded ✅")

    # --------- Вспомогательные методы ---------

    def _resolve_device(self, mode: str) -> torch.device:
        mode = (mode or "auto").lower()
        if mode == "cpu":
            return torch.device("cpu")
        if mode == "cuda":
            if not torch.cuda.is_available():
                logger.warning("VLM_DEVICE=cuda but CUDA is not available, falling back to CPU")
                return torch.device("cpu")
            return torch.device("cuda")

        # auto
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")

    def _build_messages(self, image: Image.Image, prompt: str) -> list[Dict[str, Any]]:
        """
        Формируем сообщения в формате, который ожидает SmolVLM2
        (image + text в одном message).
        """
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

    def analyze_image(self, image_path: str, prompt: str, mode: str = "chat") -> str:
        """
        Универсальная функция анализа:
        - mode="chat"  -> VQA / captioning
        - mode="ocr"   -> OCR, используем специальный промпт
        """
        # Подготовка промпта
        if mode == "ocr":
            final_prompt = f"{config.OCR_SYSTEM_PROMPT}\n\nImage:"
        else:
            final_prompt = prompt

        image = Image.open(image_path).convert("RGB")

        messages = self._build_messages(image, final_prompt)

        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device, dtype=self.dtype)

        generated_ids = self.model.generate(
            **inputs,
            do_sample=config.GENERATION_TEMPERATURE > 0.0,
            temperature=config.GENERATION_TEMPERATURE if config.GENERATION_TEMPERATURE > 0.0 else None,
            max_new_tokens=config.MAX_NEW_TOKENS,
        )

        generated_texts = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )
        text = generated_texts[0].strip()

        # Иногда модель добавляет "Assistant: ..." — подчистим
        for marker in ("Assistant:", "assistant:"):
            if marker in text:
                text = text.split(marker, 1)[-1].strip()

        return text

    # --------- Запуск воркера ---------

    def start(self, warmup: bool = True) -> None:
        """
        Опциональный прогрев на DEMO_IMAGE и запуск фонового цикла.
        """
        if warmup:
            try:
                demo_path = config.DEMO_IMAGE
                if isinstance(demo_path, (str, Path)):
                    demo_path = Path(demo_path)
                if demo_path.exists():
                    logger.info(f"[InferenceWorker] Warmup with demo image: {demo_path}")
                    _ = self.analyze_image(str(demo_path), "Quick warmup", mode="chat")
                    logger.info("[InferenceWorker] Warmup done ✅")
                else:
                    logger.info("[InferenceWorker] Warmup skipped: demo image not found")
            except Exception as e:
                logger.warning(f"[InferenceWorker] Warmup failed (not critical): {e}")

        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()
        logger.info("[InferenceWorker] Worker thread started ✅")

    def _loop(self) -> None:
        logger.info("[InferenceWorker] Waiting for tasks...")
        while True:
            task: Dict[str, Any] = self.task_queue.get()
            try:
                task_id = task.get("id")
                image_path = task["image_path"]
                prompt = task.get("prompt", "")
                mode = task.get("mode", "chat")

                logger.info(f"[InferenceWorker] Processing task_id={task_id}, mode={mode}")
                result_text = self.analyze_image(image_path, prompt, mode=mode)

                self.result_queue.put(
                    {"id": task_id, "result": result_text}
                )
            except Exception as e:
                logger.exception("[InferenceWorker] Error while processing task")
                self.result_queue.put(
                    {"id": task.get("id"), "error": str(e)}
                )
            finally:
                self.task_queue.task_done()
