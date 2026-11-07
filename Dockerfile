# Use your optimized, explicit Python runtime base image
FROM python:3.11-slim

# Prevent Python from writing pyc files to disk and ensure logs dump instantly
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

# Install basic system layer dependencies required for C-extensions and compilers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency structures first to leverage Docker layer caching speeds
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Create a local cache directory for SentenceTransformers to avoid permission blocks in the cloud
RUN mkdir -p /.cache && chmod -R 777 /.cache
ENV HF_HOME=/.cache

# Copy the entire workspace layout securely into the working directory
COPY . .

# Hugging Face Spaces strictly requires exposing port 7860
EXPOSE 7860

# Launch Streamlit directly as the primary service interface bound to HF rules
CMD ["streamlit", "run", "ui.py", "--server.port=7860", "--server.address=0.0.0.0"]