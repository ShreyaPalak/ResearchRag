import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL_NAME

_model = None


def _get_model() -> SentenceTransformer:
    """Lazy-load the embedding model on first use."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Embed a batch of chunks (L2-normalized, float32) for indexing."""
    return _get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string (L2-normalized, float32) for search."""
    return _get_model().encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype(np.float32)
