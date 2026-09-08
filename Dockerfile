FROM python:3.14-slim

# No system dependencies needed — pypdfium2 bundles the PDFium binary in its wheel.

WORKDIR /app

# Copy requirements first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (excluded via .dockerignore: .env, data/, __pycache__, etc.)
COPY . .

# Create directories for persistence
RUN mkdir -p /app/reports_input /app/data

# Environment variables — Streamlit config moved to ENV so --server.* flags aren't duplicated
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
# Disable file watcher in Docker — no point watching for changes inside a container.
# NOTE: env var name is derived from the config key "server.fileWatcherType"
# (see streamlit/config_option.py), so it must be STREAMLIT_SERVER_FILE_WATCHER_TYPE.
ENV STREAMLIT_SERVER_FILE_WATCHER_TYPE=none

EXPOSE 8501

# Health check so orchestrators can detect crashes
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py"]
