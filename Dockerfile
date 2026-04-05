# SelfHealRL — OpenEnv Docker image
# Builds the FastAPI environment server on port 8000
#
# Build:  docker build -t selfheal-rl .
# Run:    docker run -p 8000:8000 selfheal-rl
# Test:   curl http://localhost:8000/health

FROM python:3.10-slim

# ── System deps ───────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────
WORKDIR /app

# ── Python deps (cached layer) ────────────────────────────────
COPY requirements.txt .

# Install CPU-only torch first (much smaller image, ~800MB vs 3GB)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining deps
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy project files ────────────────────────────────────────
COPY config.py .
COPY models.py .
COPY openenv.yaml .
COPY run.py .
COPY core/   core/
COPY env/    env/
COPY server/ server/
COPY training/ training/
COPY models/ models/

# ── Environment ───────────────────────────────────────────────
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# ── Expose port ───────────────────────────────────────────────
EXPOSE 8000

# ── Healthcheck ───────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ── Entrypoint ────────────────────────────────────────────────
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
