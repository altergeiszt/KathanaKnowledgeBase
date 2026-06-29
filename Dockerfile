FROM python:3.11-slim

# System deps: build tools for datasketch/sentence-transformers native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache friendly)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Clone LightRAG and patch __init__.py to register SurrealDB adapter.
# We do this at build time so the image is self-contained.
RUN pip install --no-cache-dir lightrag-hku && \
    python /app/patch_lightrag.py

# Copy application source
COPY surrealdb_impl.py /app/
COPY patch_lightrag.py /app/
COPY ingest.py         /app/
COPY api.py            /app/

# Re-run the patch (pip install may have reset the file)
RUN python /app/patch_lightrag.py

# Default: run the API server.
# Override CMD in docker-compose for the ingest worker.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
