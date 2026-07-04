import os

# --- Groq ---
# Read the key but don't crash at import time — validated at usage in rag_engine.
GROQ_API_KEY = os.environ.get('GROQ_API_KEY', '')

GROQ_MODEL = os.environ.get('GROQ_MODEL', 'openai/gpt-oss-20b')

# --- Embeddings ---
EMBEDDING_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'

# --- Chunking ---
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# --- Retrieval ---
TOP_K = 4

# --- Server ---
PORT = int(os.environ.get('PORT', 7860))
