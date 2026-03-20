FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (Docker layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .
COPY src/ ./src/
COPY data/barras/ ./data/barras/
COPY data/exogenous/ ./data/exogenous/
COPY scripts/ ./scripts/

# Set environment variables
ENV PORT=8080
ENV PYTHONUNBUFFERED=1
ENV USE_BIGQUERY=true
ENV GCP_PROJECT_ID=geologgia-map
ENV BQ_DATASET=cmarg

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8080/ || exit 1

# Run with gunicorn for production
CMD exec gunicorn app:server \
    --bind 0.0.0.0:$PORT \
    --workers 2 \
    --threads 4 \
    --timeout 300 \
    --graceful-timeout 120 \
    --keep-alive 5 \
    --log-level info
