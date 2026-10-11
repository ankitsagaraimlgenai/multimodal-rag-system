FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HF_HOME=/models PYTHONPATH=/app
WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel.
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

# Bake CLIP into the image so containers start without internet access.
RUN python -c "from transformers import CLIPModel, CLIPProcessor as P; n='openai/clip-vit-base-patch32'; CLIPModel.from_pretrained(n); P.from_pretrained(n)"

COPY . .
RUN useradd -m -u 1000 app && mkdir -p /app/data && chown -R app:app /app /models
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/healthz').status==200 else 1)"

# One worker on purpose: the vector index lives in memory and one process owns the ingestion lock.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
