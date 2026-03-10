# ── Stage 1: Build React Frontend ──────────────────────────
FROM node:20-slim AS frontend-build
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci
COPY frontend/ .
RUN npm run build

# ── Stage 2: Python Backend + Serve ───────────────────────
FROM python:3.11-slim

# Reduce TF noise & memory
ENV TF_CPP_MIN_LOG_LEVEL=2
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2

WORKDIR /app

# Install Python deps first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend code
COPY server.py .
COPY saved_models/ saved_models/

# Copy built frontend from stage 1
COPY --from=frontend-build /app/frontend/dist frontend/dist

EXPOSE 10000

CMD ["gunicorn", "server:app", "--bind", "0.0.0.0:10000", "--workers", "1", "--threads", "2", "--timeout", "120"]
