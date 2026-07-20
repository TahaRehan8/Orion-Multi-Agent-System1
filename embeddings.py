"""
Orion — Embeddings Module
Upgraded to all-mpnet-base-v2 (768-dim) for significantly better
semantic retrieval quality over all-MiniLM-L6-v2 (384-dim).
Exposes both single and batch encoding.
"""
from sentence_transformers import SentenceTransformer

# Singleton — loaded once at import time
_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """Encode a single text string into a float vector."""
    return _model.encode(text, show_progress_bar=False).tolist()


def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts in batches.
    batch_size=64 balances VRAM/RAM usage vs. throughput.
    """
    return _model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).tolist()
