FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# torch CPU-only (Koyeb라 GPU 안 씀)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs

# 모델 파일만 포함
COPY outputs/models ./outputs/models

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

CMD ["python", "-m", "src.infer_api"]
