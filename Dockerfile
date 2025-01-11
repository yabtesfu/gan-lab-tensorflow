# Slim CPU-only image for the GAN Observatory.
# Note: it installs only the ".[web]" extra -- NumPy + FastAPI + Uvicorn --
# NOT TensorFlow. The live 2D backend is pure NumPy, so the image stays small
# and builds in seconds.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir ".[web]"

ENV HOST=0.0.0.0 \
    PORT=8000
EXPOSE 8000

# Basic liveness probe target is /healthz.
CMD ["python", "-m", "gan_lab_tensorflow.live.server"]
