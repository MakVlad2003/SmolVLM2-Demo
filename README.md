Окей, давай просто нормальный, ровный `README.md` без лишней воды. Вот готовый текст — можешь копировать как есть в файл.
# SmolVLM2 Demo — Vision Chat & OCR 🖼️💬

This repository provides a **Dockerized web demo** for the  
[SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct) model.

The app includes:

- **Vision Chat (VQA / Captioning)** — multi-turn chat about a single image  
- **OCR (Text recognition)** — extract text from an image and download result as `.txt`

It is designed to:

- run **fully inside a Docker container** (CPU or GPU),
- work **offline** after the first model download,
- keep **model weights on the host** via a mounted Hugging Face cache directory.

> ⚠️ This is a demo / educational project, not a production-hardened system.

---

## Features

- 🧠 SmolVLM2 500M vision-language model (model size is configurable)
- 💬 Vision Chat: image captioning & visual question answering in a chat-like interface
- 📝 OCR: recognize text from images and save it as `.txt`
- 🔁 Multi-turn dialogue on the same image (no need to re-upload the image)
- 📥 Download last model answer (chat) and OCR result as text files
- 🧮 CPU **and** GPU (CUDA) execution modes
- 💾 Hugging Face cache and model weights stored on the host (via Docker volume)
- 🐳 Single container with **web UI** and **HTTP API**

---

## Requirements

- Docker (and optionally Docker Compose v2)
- For GPU mode:
  - NVIDIA GPU with recent drivers
  - `--gpus all` support in Docker
  - At least **4 GB VRAM** recommended for the 500M model

---

## Quick Start

### 1. Build the Docker image

From repository root:

```bash
docker build -t vlm-demo:latest .
````

### 2. Run on GPU

```bash
docker run --rm --gpus all \
  -p 8888:8888 \
  -e VLM_MODEL_SIZE=500M \
  -e VLM_DEVICE=cuda \
  -e VLM_PORT=8888 \
  -e HF_LOCAL_ONLY=0 \
  -e HF_HOME=/data/hf-cache \
  -e HUGGINGFACE_HUB_CACHE=/data/hf-cache \
  -e TRANSFORMERS_CACHE=/data/hf-cache \
  -v "$(pwd)/hf-cache:/data/hf-cache" \
  vlm-demo:latest
```

Then open in your browser:

> [http://localhost:8888/ui](http://localhost:8888/ui)

The first run will download model weights into `./hf-cache` on the host.

### 3. Run on CPU

```bash
docker run --rm \
  -p 8888:8888 \
  -e VLM_MODEL_SIZE=500M \
  -e VLM_DEVICE=cpu \
  -e VLM_PORT=8888 \
  -e HF_LOCAL_ONLY=1 \
  -e HF_HOME=/data/hf-cache \
  -e HUGGINGFACE_HUB_CACHE=/data/hf-cache \
  -e TRANSFORMERS_CACHE=/data/hf-cache \
  -v "$(pwd)/hf-cache:/data/hf-cache" \
  vlm-demo:latest
```

CPU mode is slower but does not require a GPU.

---

## Docker Compose (v2)

A `docker-compose.yml` is provided.

Example:

```bash
VLM_MODEL_SIZE=500M VLM_DEVICE=cuda HF_LOCAL_ONLY=0 VLM_PORT=8888 docker compose up
```

Then open:

> [http://localhost:8888/ui](http://localhost:8888/ui)

Stop the stack:

```bash
docker compose down
```

You can change variables in the command or in `docker-compose.yml`.

---

## Configuration

The container is configured through environment variables:

| Variable                | Required | Default          | Description                                                                |
| ----------------------- | -------- | ---------------- | -------------------------------------------------------------------------- |
| `VLM_MODEL_SIZE`        | yes      | `500M`           | SmolVLM2 model size (e.g. `500M`, other sizes if supported by backend).    |
| `VLM_DEVICE`            | yes      | `cpu`            | `cuda` for GPU, `cpu` for CPU inference.                                   |
| `VLM_PORT`              | yes      | `8888`           | HTTP port inside the container.                                            |
| `HF_LOCAL_ONLY`         | yes      | `1`              | `1` → use only local cache; `0` → allow downloading from Hugging Face Hub. |
| `HF_HOME`               | yes      | `/data/hf-cache` | Base Hugging Face cache directory.                                         |
| `HUGGINGFACE_HUB_CACHE` | yes      | `/data/hf-cache` | HF Hub cache path.                                                         |
| `TRANSFORMERS_CACHE`    | yes      | `/data/hf-cache` | Transformers cache path.                                                   |

Port mapping is controlled by Docker:

```bash
-p HOST_PORT:CONTAINER_PORT  # e.g. -p 9000:9000 with VLM_PORT=9000
```

Then UI will be at `http://localhost:HOST_PORT/ui`.

---

## Model Cache & Offline Mode

Model weights and HF cache are kept on the host using a bind mount:

* Inside container: `/data/hf-cache`
* On host: e.g. `./hf-cache`

Mount example:

```bash
-v "$(pwd)/hf-cache:/data/hf-cache"
```

### First (online) run

Use `HF_LOCAL_ONLY=0` so the container can download model weights into the cache.

### Offline runs

After weights are cached, you can run completely offline:

```bash
-e HF_LOCAL_ONLY=1
```

As long as the required model files are present in the mounted directory, the
container will start without any network access.

---

## Web UI

The web interface is served at:

> `http://<host>:<port>/ui` (default: `http://localhost:8888/ui`)

The UI has two tabs:

### 1. Vision Chat (VQA / Captioning)

Use cases:

* image captioning
* visual question answering
* general image description / reasoning

Workflow:

1. **Upload an image** in the *Image for analysis* panel.
2. Enter a prompt in *Your question / instruction*.
3. Press **Enter** or click **Send**.
4. The model’s reply is appended to the **chat history**.
5. The field *Download last answer (.txt)* lets you save the latest reply as a `.txt` file.

Validation:

* If you try to send a message without an image, the assistant responds with
  a clear warning and no model inference is triggered.

The same image can be used for multiple questions without re-uploading.

### 2. OCR (Text recognition)

Use case:

* recognize text from images and export to `.txt`.

Workflow:

1. Switch to **“OCR (Text recognition)”** tab.
2. **Upload an image** with text.
3. Click **Run OCR**.
4. Recognized text is displayed in the right panel.
5. Use *Download result (.txt)* to save it.

Validation:

* If you run OCR without an image, a user-friendly warning is shown.

---

## HTTP API

The backend also exposes a simple HTTP endpoint similar to the original SmolVLM demo.

### POST `/ptt/convert`

* **Method:** `POST`
* **Content type:** `multipart/form-data`
* **Fields:**

  * `query` — text prompt / question
  * `image` — optional image file

Example with image:

```bash
curl -X POST "http://localhost:8888/ptt/convert" \
  -H "Accept: application/json" \
  -F "query=What is in this image?" \
  -F "image=@cat.jpg;type=image/jpeg"
```

Example without image (backend may use a demo image or return an error,
depending on configuration):

```bash
curl -X POST "http://localhost:8888/ptt/convert" \
  -F "query=Describe the demo image."
```

---

## Project Structure (high level)

```text
app/
  ├─ main.py          # FastAPI / Uvicorn entrypoint, mounts UI and API
  ├─ ui.py            # Gradio UI (Vision Chat + OCR tabs)
  ├─ inference.py     # SmolVLM2 loading and inference worker
  ├─ result_broker.py # Simple in-memory result broker for async tasks
  ├─ config.py        # Reads environment variables (device, model id, port, etc.)
  └─ ...
Dockerfile
docker-compose.yml
README.md
```

---

## Model Reference

This demo uses **SmolVLM2** from Hugging Face:

* Blog post / technical overview:
  [https://huggingface.co/blog/smolvlm2](https://huggingface.co/blog/smolvlm2)
* Model card:
  [https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct](https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct)

Please refer to the model card for licensing and intended use.

---

## License

This project is released under the [MIT License](LICENSE).

The underlying models are provided by
[HuggingFaceTB](https://huggingface.co/HuggingFaceTB) and keep their own licenses.
