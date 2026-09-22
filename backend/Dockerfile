FROM python:3.12-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app:/app/backend

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy both backend and ai_engine source trees for direct access
COPY backend /app/backend
COPY ai_engine /app/ai_engine

WORKDIR /app/backend

# Ensure persistent staging directory exists
RUN mkdir -p storage/staging/reels

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Production entrypoint without auto-reload
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
