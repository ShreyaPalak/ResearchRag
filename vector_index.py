import faiss
import numpy as np


class VectorIndex:
    """Wrapper around a FAISS flat inner-product index.

    Since embeddings are L2-normalized, inner product == cosine similarity.
    """

    def __init__(self):
        self.index = None
        self.chunks: list[str] = []
        self.chunk_pages: list[int] = []

    def build(self, embeddings: np.ndarray, chunks: list[str], chunk_pages: list[int]) -> None:
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        self.chunks = chunks
        self.chunk_pages = chunk_pages

    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def search(self, query_vec: np.ndarray, k: int) -> list[dict]:
        if not self.is_ready():
            return []
        scores, indices = self.index.search(query_vec, k)
        return [
            {'text': self.chunks[idx], 'page': self.chunk_pages[idx], 'score': float(score)}
            for idx, score in zip(indices[0], scores[0])
            if idx != -1
        ]
