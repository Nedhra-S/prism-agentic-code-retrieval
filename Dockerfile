FROM python:3.12-slim

WORKDIR /app

# System dependencies used by the Python/ML stack.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install the exact dependency versions used by the project.
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy project source and supporting files.
COPY . .

# Default container command: run the lightweight sanity check.
# To run the official evaluation instead:
# docker run <image> python eval/run_mteb.py
CMD ["python", "eval/quick_check.py"]