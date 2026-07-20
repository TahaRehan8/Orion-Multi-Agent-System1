"""
Orion — Embeddings Module
Switched to Hugging Face Inference API for all-MiniLM-L6-v2 (384-dim).
This removes PyTorch dependencies and massive RAM overhead,
allowing the application to run smoothly on free server tiers.
"""
import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
HF_TOKEN = os.getenv("HF_TOKEN")

headers = {}
if HF_TOKEN:
    headers["Authorization"] = f"Bearer {HF_TOKEN}"

def _query_api(payload, retries=3):
    """Query the Hugging Face API with basic retry logic for cold starts."""
    for attempt in range(retries):
        try:
            response = httpx.post(API_URL, headers=headers, json=payload, timeout=30.0)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 503:
                # Model is loading, wait and retry
                time.sleep(2 * (attempt + 1))
                continue
            else:
                response.raise_for_status()
        except httpx.RequestError as e:
            if attempt == retries - 1:
                raise e
            time.sleep(2)
    
    # If we exhaust retries and still didn't get 200
    response = httpx.post(API_URL, headers=headers, json=payload, timeout=30.0)
    response.raise_for_status()
    return response.json()


def get_embedding(text: str) -> list[float]:
    """Encode a single text string into a float vector."""
    result = _query_api({"inputs": [text]})
    if isinstance(result, list) and len(result) > 0:
        return result[0]
    return []


def get_embeddings_batch(texts: list[str], batch_size: int = 64) -> list[list[float]]:
    """
    Encode a list of texts in batches via the API.
    """
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        result = _query_api({"inputs": batch})
        if isinstance(result, list):
            all_embeddings.extend(result)
        else:
            raise ValueError(f"Unexpected API response type: {type(result)}")
    return all_embeddings

