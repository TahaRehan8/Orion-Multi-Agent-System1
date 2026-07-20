"""
Orion — Embeddings Module
Using fastembed for all-MiniLM-L6-v2 (384-dim).
This runs locally using ONNX, bypassing network issues
and avoiding heavy PyTorch RAM overhead!
"""
from fastembed import TextEmbedding

# Singleton — loaded once at import time. 
# It automatically downloads the ONNX model (if not cached) and runs it efficiently.
_model = TextEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")


def get_embedding(text: str) -> list[float]:
    """Encode a single text string into a float vector."""
    # fastembed returns a generator of numpy arrays, we want the first one as a list
    result = list(_model.embed([text]))
    if result:
        return result[0].tolist()
    return []


def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts in batches.
    fastembed handles batching natively under the hood.
    """
    results = list(_model.embed(texts, batch_size=batch_size))
    return [vec.tolist() for vec in results]

