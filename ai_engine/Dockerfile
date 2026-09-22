FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install AI Engine dependencies
COPY ai_engine/rag/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy ai_engine source tree
COPY ai_engine /app/ai_engine

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Production entrypoint without auto-reload
CMD ["uvicorn", "ai_engine.rag.server:app", "--host", "0.0.0.0", "--port", "8001"]
