import os
from pathlib import Path

# Доступные варианты моделей SmolVLM2 (video instruct)
SMOLVLM2_MODELS = {
    "256M": "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "500M": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
}

# --- Размер модели: 256M или 500M ---
MODEL_SIZE = os.getenv("VLM_MODEL_SIZE", "500M")
if MODEL_SIZE not in SMOLVLM2_MODELS:
    raise ValueError(
        f"Invalid VLM_MODEL_SIZE={MODEL_SIZE!r}. "
        f"Use one of: {', '.join(SMOLVLM2_MODELS.keys())}"
    )

MODEL_ID = SMOLVLM2_MODELS[MODEL_SIZE]

# --- Режим устройства: "auto" | "cuda" | "cpu" ---
DEVICE_MODE = os.getenv("VLM_DEVICE", "auto").lower()

# --- Порт FastAPI/Gradio ---
PORT = int(os.getenv("VLM_PORT", "8888"))

# --- Каталог для кэша/весов моделей (монтируем volume сюда) ---
# По умолчанию просто /data/hf-cache, но Hugging Face будет
# уважать HF_HOME / HUGGINGFACE_HUB_CACHE, если они выставлены.
MODEL_CACHE_DIR = Path(os.getenv("VLM_MODEL_CACHE", "/data/hf-cache"))

# --- Базовая директория app ---
BASE_DIR = Path(__file__).resolve().parent

# Демонстрационное изображение (потом положим файл сюда)
DEMO_IMAGE = BASE_DIR / "images" / "cat.jpg"

# --- Настройки генерации текста ---
MAX_NEW_TOKENS = int(os.getenv("VLM_MAX_NEW_TOKENS", "256"))
GENERATION_TEMPERATURE = float(os.getenv("VLM_TEMPERATURE", "0.0"))  # 0 = greedy

# --- OCR промпт ---
OCR_SYSTEM_PROMPT = (
    "You are an OCR engine. Read ALL legible text from the image and "
    "output only the extracted text, preserving line breaks when helpful. "
    "Do not add explanations, comments, or extra words."
)

# Тайм-аут ожидания ответа модели (для API/UI) в секундах
INFERENCE_TIMEOUT = int(os.getenv("VLM_INFERENCE_TIMEOUT", "120"))
