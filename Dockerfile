FROM python:3.11-slim

# ---- Environment ----
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# HuggingFace cache (kept inside container)
ENV HF_HOME=/app/.cache/huggingface
ENV TRANSFORMERS_CACHE=/app/.cache/huggingface

WORKDIR /app

# ---- System deps (minimal) ----
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# ---- Install Python deps ----
COPY requirements.txt /app/requirements.txt

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

# ---- Copy app ----
COPY . /app

# ---- Expose ----
EXPOSE 8080

# ---- Run Streamlit ----
CMD ["streamlit", "run", "chat_english.py", \
     "--server.address", "0.0.0.0", \
     "--server.port", "8080", \
     "--server.headless", "true", \
     "--browser.gatherUsageStats", "false"]
