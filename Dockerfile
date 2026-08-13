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
COPY static/ static/
COPY models/preprocessor.joblib models/preprocessor.joblib

# Render assigns a port dynamically via the $PORT env var at container
# start (unlike HF Spaces' fixed 7860), so EXPOSE here is documentation
# only -- the default (8000) is just for local `docker run` without $PORT
# set; Render always provides its own value at runtime.
EXPOSE 8000

# Exec-form CMD (a JSON array) does NOT go through a shell, so it can't
# expand $PORT -- Docker would pass the literal string "$PORT" to uvicorn
# and it would fail trying to bind a port literally named that. Using
# `sh -c` here makes the substitution actually happen at container start,
# with a fallback to 8000 for local runs where $PORT isn't set.
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
