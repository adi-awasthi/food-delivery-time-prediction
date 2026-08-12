FROM python:3.13-slim

WORKDIR /app

# LightGBM's prebuilt wheel dynamically links against libgomp (OpenMP) for
# multi-threading, which python:3.13-slim doesn't include by default --
# without this, model loading fails with an opaque
# "libgomp.so.1: cannot open shared object file" error at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY params.yaml .
COPY src/ src/
COPY models/preprocessor.joblib models/preprocessor.joblib

# Hugging Face Spaces' Docker SDK requires the container to listen on
# 0.0.0.0:7860 -- a fixed convention, not a dynamic $PORT env var.
EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
