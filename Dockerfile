FROM python:3.11-slim

WORKDIR /app

# System deps some ML wheels need for building; slim image doesn't ship these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# CPU-only torch keeps the image small; drop --index-url if you need GPU.
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Default: run the sanity check. Override with `docker run <image> python eval/run_mteb.py`
CMD ["python", "eval/quick_check.py"]
