FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 1. Copy requirements and install deps (cached layer — only re-runs if requirements.txt changes)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy application source BEFORE running the patch
#    (patch_lightrag.py needs surrealdb_impl.py to be present)
COPY surrealdb_impl.py .
COPY patch_lightrag.py .
COPY ingest.py .
COPY api.py .

# 3. Patch LightRAG — copies surrealdb_impl.py into lightrag/kg/ and
#    registers the four storage classes in lightrag/kg/__init__.py
RUN python patch_lightrag.py

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
