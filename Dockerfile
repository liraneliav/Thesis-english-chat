
---

## `Dockerfile`

```dockerfile
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PORT=8080

# hnswlib may compile; include build deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first (better caching)
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Copy app files
COPY . /app

EXPOSE 8080

CMD ["bash", "-lc", "streamlit run chat_english.py --server.address 0.0.0.0 --server.port ${PORT} --server.headless true --browser.gatherUsageStats false"]