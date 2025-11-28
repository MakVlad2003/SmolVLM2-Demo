"""
init_downloads.py

Скрипт для ПРЕДзагрузки моделей SmolVLM2 в кэш Hugging Face / примонтированный volume.

Пример использования (позже опишем в README):

    docker run --rm \
      -e HF_HOME=/data/hf-cache \
      -v /path/on/host/hf-cache:/data/hf-cache \
      your-image-name \
      python -m app.init_downloads
"""

import os
from typing import List

from huggingface_hub import snapshot_download


# Те же модели, что и в config.SMOLVLM2_MODELS
MODELS: List[str] = [
    "HuggingFaceTB/SmolVLM2-256M-Video-Instruct",
    "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
]

# Можно не тянуть лишние форматы (onnx, tflite, tf и т.п.)
IGNORE_PATTERNS = [
    "*.md",
    "*.onnx",
    "*.h5",
    "*.tflite",
    "rust_model*",
    "*.msgpack",
    "tf_model*",
]


def main() -> None:
    local_only_env = os.getenv("HF_LOCAL_ONLY", "0")
    local_files_only = local_only_env == "1"

    print(
        f"[init_downloads] Starting prefetch "
        f"(HF_LOCAL_ONLY={local_only_env}, local_files_only={local_files_only})"
    )

    for repo_id in MODELS:
        print(f"[init_downloads] Prefetching {repo_id} ...")
        snapshot_download(
            repo_id=repo_id,
            local_files_only=local_files_only,
            ignore_patterns=IGNORE_PATTERNS,
        )

    print("[init_downloads] Done ✅")


if __name__ == "__main__":
    main()
