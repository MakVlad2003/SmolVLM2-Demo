FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    HF_HOME=/data/hf-cache \
    HUGGINGFACE_HUB_CACHE=/data/hf-cache \
    TRANSFORMERS_CACHE=/data/hf-cache \
    VLM_MODEL_SIZE=500M \
    VLM_DEVICE=auto \
    VLM_PORT=8888 \
    HF_LOCAL_ONLY=0

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

EXPOSE 8888

CMD ["bash", "-lc", "uvicorn app.main:app --host 0.0.0.0 --port ${VLM_PORT}"]
