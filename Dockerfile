# 1) 가벼운 Python 이미지
FROM python:3.10-slim

# 2) 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 3) 작업 디렉토리 설정
WORKDIR /app

# 4) requirements 먼저 복사 → 캐시 최적화
COPY requirements.txt .

# 5) torch CPU-only 설치 (용량 1/5로 감소)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# 6) 나머지 패키지 설치
RUN pip install --no-cache-dir -r requirements.txt

# 7) 프로젝트 전체 복사 (불필요 파일은 .dockerignore에서 제외)
COPY . .

# 8) FastAPI + Uvicorn 실행
CMD ["uvicorn", "src.infer_api:app", "--host", "0.0.0.0", "--port", "8000"]
