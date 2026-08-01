FROM python:3.11-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# PyTorch CPU-only — separate layer so it stays cached when requirements.txt changes
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

COPY . .

CMD ["python", "finetune_yolov8_weapon_data2.py"]
