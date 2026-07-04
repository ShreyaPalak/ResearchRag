FROM python:3.11-slim

WORKDIR /app

# build-essential needed for some sentence-transformers/faiss wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Single pip install step. --extra-index-url makes torch resolve to the
# CPU-only build from PyTorch's own index, preventing sentence-transformers
# from pulling the multi-GB CUDA-enabled torch from PyPI.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --default-timeout=100 \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

COPY . .

EXPOSE 7860
ENV PORT=7860

# Pass secrets at runtime:
#   docker run -e GROQ_API_KEY=gsk_... -p 7860:7860 rag-app
CMD ["python", "app.py"]
