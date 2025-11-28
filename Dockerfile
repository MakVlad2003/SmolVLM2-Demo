# Базовый образ с PyTorch + CUDA
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Рабочая директория внутри контейнера
WORKDIR /app

# Переменные окружения:
# - кэш Hugging Face в /data/hf-cache (монтируем volume сюда)
# - базовые настройки приложения
ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    HUGGINGFACE_HUB_CACHE=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    VLM_MODEL_SIZE=500M \
    VLM_DEVICE=auto \
    VLM_PORT=8888 \
    HF_LOCAL_ONLY=0

# Устанавливаем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код приложения
COPY app ./app

# Порт, на котором слушает uvicorn
EXPOSE 8888

# Запуск приложения
# Порт берём из VLM_PORT
CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${VLM_PORT}"]
