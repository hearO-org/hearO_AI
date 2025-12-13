FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY outputs ./outputs

EXPOSE 8000

CMD ["uvicorn", "src.infer_api:app", "--host", "0.0.0.0", "--port", "8000"]
