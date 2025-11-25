FROM python:3.10-slim

# FFmpeg (librosa, audioread 등에서 필요)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 의존성 먼저 설치 (torch/torchaudio는 따로 설치하므로 requirements.txt에서는 제외)
COPY requirements.txt .

# CPU 전용 PyTorch + Torchaudio (EC2 t3.micro라 GPU 없음)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
        torch==2.5.0+cpu \
        torchaudio==2.5.0+cpu \
        --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# 애플리케이션 코드/설정/모델 파일 복사
COPY src ./src
COPY configs ./configs
COPY outputs/models ./outputs/models

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# FastAPI infer API 실행
CMD ["python", "-m", "src.infer_api"]
