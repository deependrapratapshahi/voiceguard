FROM python:3.11-slim

# System deps for librosa/soundfile audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt

# ML deps (torch/transformers) are heavy and optional -- baseline mode
# works without them. Install the lighter requirements set by default;
# set INSTALL_PRETRAINED=true at build time to include torch/transformers.
ARG INSTALL_PRETRAINED=false
RUN pip install --no-cache-dir -r requirements.txt || \
    (grep -v -E "^(torch|torchaudio|transformers)" requirements.txt > requirements.lite.txt && \
     pip install --no-cache-dir -r requirements.lite.txt)

COPY backend/ ./backend/
COPY ml/ ./ml/

ENV PYTHONPATH=/app/backend:/app
WORKDIR /app/backend

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
