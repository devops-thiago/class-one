# Production Dockerfile for ClassOne System 1 Service
FROM python:3.11-slim

# Security: Create non-root system user and group
RUN groupadd -r appgroup && useradd -r -g appgroup -u 1001 -m -d /home/appuser appuser

WORKDIR /app

# Install minimal OS dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy build files and application source before package installation
COPY pyproject.toml README.md /app/
COPY src/ /app/src/

# Install dependencies and Jev package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Create checkpoints directory and set ownership
RUN mkdir -p /app/checkpoints && chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

# Expose FastAPI service port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Environment variables
ENV PYTHONUNBUFFERED=1 \
    CLASSONE_BASE_MODEL=standalone

# Run FastAPI production server
CMD ["uvicorn", "classone.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
