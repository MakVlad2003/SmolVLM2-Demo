from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import gradio as gr

from .result_broker import ResultBroker
from . import config

Message = Dict[str, Any]
History = List[Message]


class GradioUI:
    def __init__(
        self,
        task_queue: "queue.Queue[Dict[str, Any]]",
        result_broker: ResultBroker,
    ) -> None:
        self.task_queue = task_queue
        self.result_broker = result_broker
        self._task_id_counter = 0

    def _next_task_id(self) -> int:
        self._task_id_counter += 1
        return self._task_id_counter

    def chat_infer(
        self,
        image_path: Optional[str],
        history: Optional[History],
        user_message: str,
    ) -> Tuple[History, str, Optional[str]]:
        if history is None:
            history = []

        user_message = (user_message or "").strip()

        if not user_message:
            return history, "", None

        if not image_path:
            history.append(
                {"role": "assistant", "content": "Please upload an image first."}
            )
            return history, "", None

        history.append({"role": "user", "content": user_message})
        history.append({"role": "assistant", "content": "…"})

        task_id = self._next_task_id()
        self.task_queue.put(
            {
                "id": task_id,
                "image_path": image_path,
                "prompt": user_message,
                "mode": "chat",
            }
        )

        waiter = self.result_broker.register(task_id)

        try:
            result = waiter.get(timeout=config.INFERENCE_TIMEOUT)
        except queue.Empty:
            history[-1]["content"] = (
                "Inference timeout exceeded. Please try again."
            )
            return history, "", None

        if "error" in result:
            history[-1]["content"] = f"Error during processing: {result['error']}"
            answer_text = history[-1]["content"]
        else:
            answer = (result.get("result") or "").strip()
            if not answer:
                answer = "(model returned an empty answer)"
            history[-1]["content"] = answer
            answer_text = answer

        ts = int(time.time())
        out_dir = Path("chat_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"chat_result_{ts}.txt"
        out_path.write_text(answer_text, encoding="utf-8")

        return history, "", str(out_path)

    def ocr_infer(self, image_path: Optional[str]) -> Tuple[str, Optional[str]]:
        if not image_path:
            return "Please upload an image with text.", None

        task_id = self._next_task_id()
        self.task_queue.put(
            {
                "id": task_id,
                "image_path": image_path,
                "prompt": "",
                "mode": "ocr",
            }
        )

        waiter = self.result_broker.register(task_id)

        try:
            result = waiter.get(timeout=config.INFERENCE_TIMEOUT)
        except queue.Empty:
            return "OCR timeout exceeded. Please try again.", None

        if "error" in result:
            return f"OCR error: {result['error']}", None

        text = (result.get("result") or "").strip()
        if not text:
            text = "(no text could be recognized)"

        ts = int(time.time())
        out_dir = Path("ocr_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"ocr_result_{ts}.txt"
        out_path.write_text(text, encoding="utf-8")

        return text, str(out_path)

    def build(self):
        style_html = """
        <style>
        .gradio-container {
            max-width: 1280px;
            margin: 0 auto;
        }

        /* Хедер и подписи по центру */
        .gradio-container h1,
        .gradio-container h2,
        .gradio-container h3,
        .gradio-container label {
            text-align: center;
            width: 100%;
        }

        /* Центрируем "карусель" табов */
        div[role='tablist'] {
            display: flex !important;
            justify-content: center !important;
        }

        /* Карточки */
        .card {
            border-radius: 10px;
            border: 1px solid #353535;
            padding: 10px;
            display: flex;
            flex-direction: column;
        }

        /* Основная высота карточек (чтоб не прыгали до/после загрузки) */
        .card-main {
            min-height: 520px;
        }

        /* Общий класс для рядов с двумя колонками (Vision + OCR) */
        .main-row {
            display: flex !important;
            flex-wrap: nowrap !important;
            gap: 12px;
        }
        .main-row > div {
            flex: 1 1 0 !important;
            min-width: 0 !important;
        }

        /* Внутренности карточки растягиваем по высоте */
        .card [data-testid="chatbot"],
        .card [data-testid="image"],
        .card textarea {
            flex: 1;
        }
        </style>
        """

        with gr.Blocks(title="SmolVLM2 Demo") as demo:
            gr.HTML(style_html)

            gr.Markdown(
                f"""
<div style="text-align: center">

# SmolVLM Chat

**Model:** `{config.MODEL_ID}`  
**Device mode:** `{config.DEVICE_MODE}` • **Default port:** `{config.PORT}`

</div>
"""
            )

            with gr.Tab("Vision Chat (VQA / Captioning)"):
                with gr.Row(equal_height=True, elem_classes=["main-row"]):
                    with gr.Column():
                        with gr.Group(elem_classes=["card", "card-main"]):
                            gr.Markdown("### Chatbot")
                            chat_history = gr.Chatbot(
                                label="",
                                height=460,   
                            )

                    with gr.Column():
                        with gr.Group(elem_classes=["card", "card-main"]):
                            gr.Markdown("### Uploaded image")
                            chat_image = gr.Image(
                                label="Image for analysis",
                                type="filepath",
                                height=460,   
                            )

                with gr.Row():
                    chat_input = gr.Textbox(
                        label="Your question / instruction",
                        placeholder="Ask the model about the image...",
                        lines=2,
                    )

                with gr.Row():
                    with gr.Column():
                        send_btn = gr.Button("Send")
                    with gr.Column():
                        chat_file = gr.File(label="Download last answer (.txt)")

                def chat_wrapper(image, history, message):
                    new_history, cleared, file_path = self.chat_infer(
                        image, history, message
                    )
                    return new_history, cleared, file_path

                send_btn.click(
                    fn=chat_wrapper,
                    inputs=[chat_image, chat_history, chat_input],
                    outputs=[chat_history, chat_input, chat_file],
                    api_name=False,
                )

                chat_input.submit(
                    fn=chat_wrapper,
                    inputs=[chat_image, chat_history, chat_input],
                    outputs=[chat_history, chat_input, chat_file],
                    api_name=False,
                )

            with gr.Tab("OCR (Text recognition)"):
                with gr.Row(equal_height=True, elem_classes=["main-row"]):
                    with gr.Column():
                        with gr.Group(elem_classes=["card", "card-main"]):
                            gr.Markdown("### Image with text")
                            ocr_image = gr.Image(
                                label="Upload image with text",
                                type="filepath",
                                height=460,
                            )

                    with gr.Column():
                        with gr.Group(elem_classes=["card", "card-main"]):
                            gr.Markdown("### Recognized text")
                            ocr_text = gr.Textbox(
                                label="",   
                                lines=18,
                            )

                with gr.Row():
                    with gr.Column():
                        ocr_button = gr.Button("Run OCR")
                    with gr.Column():
                        ocr_file = gr.File(label="Download result (.txt)")

                def ocr_wrapper(image):
                    return self.ocr_infer(image)

                ocr_button.click(
                    fn=ocr_wrapper,
                    inputs=[ocr_image],
                    outputs=[ocr_text, ocr_file],
                    api_name=False,
                )

        return demo
