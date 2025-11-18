FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements만 먼저 복사
COPY requirements.txt .

# torch CPU-only 설치
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# 필요한 폴더만 선택적으로 복사
COPY src ./src
COPY configs ./configs

# 만약 infer_api.py가 src 안에 있다면 이것도 OK
# 없으면 다음 추가
# COPY infer_api.py .

ENV PYTHONUNBUFFERED=1 PYTHONPATH=/app

CMD ["python", "-m", "src.infer_api"]
