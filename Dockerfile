# ── Build Stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System deps
RUN apt-get update && apt-get install -y \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App Stage ─────────────────────────────────────────────────────────────────
FROM base AS app

WORKDIR /app
COPY . .

# Create data dirs
RUN mkdir -p data/sessions data/eval_datasets reports screenshots

# Expose Streamlit port
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app

CMD ["streamlit", "run", "app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]
