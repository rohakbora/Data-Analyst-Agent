FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip/setuptools/wheel first (avoids many slow source builds)
RUN pip install --upgrade pip setuptools wheel

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install -r requirements.txt

# Set Hugging Face cache to a writable location
# Hugging Face cache (only HF_HOME, TRANSFORMERS_CACHE is deprecated)
ENV HF_HOME=/tmp/huggingface
ENV MPLCONFIGDIR=/tmp/matplotlib


# Pre-download and save model to /app/models
RUN python -c "from sentence_transformers import SentenceTransformer; \
    SentenceTransformer('all-MiniLM-L6-v2').save('/app/models/all-MiniLM-L6-v2')"


# Copy the application code
COPY . .

# Create necessary directories
RUN mkdir -p /tmp

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    FLASK_APP=app.py \
    FLASK_ENV=production

# Expose the port
EXPOSE 7860

# Run the application
CMD ["gunicorn", "main:app", "--workers", "2", "--bind", "0.0.0.0:7860", "--timeout", "300", "--keep-alive", "5", "--max-requests", "200", "--max-requests-jitter", "50", "--preload", "--access-logfile", "-", "--error-logfile", "-"]

