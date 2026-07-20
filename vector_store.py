"""
Orion — Vector Store Module (Qdrant)
Upgraded with:
- Local Qdrant persistence
- MMR-style diversity re-ranking on retrieval
"""
import os
import uuid
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct, Filter
from embeddings import get_embedding, get_embeddings_batch
from sentence_transformers import CrossEncoder

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
PERSIST_DIR = os.path.join(PROJECT_ROOT, "qdrant_db")

# Single persistent client
_client = QdrantClient(path=PERSIST_DIR)

# Ingestion batch size — balance memory vs. throughput
_INGEST_BATCH = 256

# Lazy-loaded cross-encoder model
_cross_encoder = None

def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        print("  [VectorStore] Loading Cross-Encoder model...")
        _cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    return _cross_encoder


def get_or_create_collection(name: str, vector_size: int = 1536):
    """Ensure the Qdrant collection exists."""
    if not _client.collection_exists(collection_name=name):
        _client.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )


def add_documents(
    collection_name: str,
    documents: list[str],
    ids: list[str],
    metadatas: list[dict] = None,
) -> None:
    """
    Upsert documents in batches.
    Uses batch embedding (get_embeddings_batch) for throughput.
    """
    if metadatas is None:
        metadatas = [{}] * len(documents)

    # Filter out blank documents
    filtered = [
        (doc, id_, meta)
        for doc, id_, meta in zip(documents, ids, metadatas)
        if doc and doc.strip()
    ]
    if not filtered:
        return

    docs, ids_, metas = zip(*filtered)
    docs, ids_, metas = list(docs), list(ids_), list(metas)
    
    # ---------------------------------------------------------
    # Sparse Indexing (TF-IDF)
    # ---------------------------------------------------------
    print(f"  [VectorStore] Building TF-IDF sparse index for '{collection_name}'...")
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(docs)
    
    sparse_index_path = os.path.join(PERSIST_DIR, f"{collection_name}_sparse.joblib")
    joblib.dump({
        'vectorizer': vectorizer,
        'matrix': tfidf_matrix,
        'docs': docs,
        'metas': metas
    }, sparse_index_path)
    print(f"  [VectorStore] Saved sparse index to {sparse_index_path}")
    # ---------------------------------------------------------

    # Need to know vector size to create collection if it doesn't exist.
    # Get first batch of embeddings to determine size
    first_batch_embeddings = get_embeddings_batch(docs[:1])
    if not first_batch_embeddings:
        return
    vector_size = len(first_batch_embeddings[0])
    get_or_create_collection(collection_name, vector_size=vector_size)

    # Embed + upsert in batches
    for start in range(0, len(docs), _INGEST_BATCH):
        end = start + _INGEST_BATCH
        batch_docs = docs[start:end]
        batch_ids = ids_[start:end]
        batch_metas = metas[start:end]
        
        # Only re-calculate if we didn't just calculate it above
        if start == 0 and len(batch_docs) == 1:
            batch_embeddings = first_batch_embeddings
        else:
            batch_embeddings = get_embeddings_batch(batch_docs)
            
        points = []
        for i in range(len(batch_docs)):
            # Qdrant needs UUIDs or integers
            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(batch_ids[i])))
            
            payload = batch_metas[i] if batch_metas[i] else {}
            payload["document"] = batch_docs[i]
            payload["original_id"] = str(batch_ids[i])
            
            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=batch_embeddings[i],
                    payload=payload
                )
            )
            
        _client.upsert(
            collection_name=collection_name,
            points=points
        )
        print(f"  [VectorStore] Upserted {min(end, len(docs))}/{len(docs)} docs into '{collection_name}'")


def query_collection(
    collection_name: str,
    query: str,
    top_k: int = 5,
    where: dict = None, # Qdrant filter object is expected here if filtering is needed
) -> dict:
    """
    Query a collection.
    Returns a result dict structured similarly to ChromaDB for backward compatibility.
    """
    get_or_create_collection(collection_name) # Assuming default size 1536 if empty
    query_embedding = get_embedding(query)
    
    # Simple dictionary conversion to Qdrant MatchValue filter
    # In a real app, you might want to translate dicts to Qdrant Filter objects explicitly
    qdrant_filter = None
    if where:
        from qdrant_client.models import Filter, FieldCondition, MatchValue
        conditions = [FieldCondition(key=k, match=MatchValue(value=v)) for k, v in where.items()]
        qdrant_filter = Filter(must=conditions)
        
    try:
        query_response = _client.query_points(
            collection_name=collection_name,
            query=query_embedding,
            limit=top_k,
            query_filter=qdrant_filter,
            with_payload=True
        )
        results = query_response.points
    except Exception as e:
        print(f"[VectorStore] Query Exception: {e}")
        # Collection might not be initialized with vectors yet
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # Format like ChromaDB: {"documents": [[doc1, doc2]], "metadatas": [[m1, m2]], "distances": [[d1, d2]]}
    # Qdrant returns score (cosine similarity), Chroma returns distance (1 - cosine similarity)
    return {
        "documents": [[res.payload.get("document", "") for res in results]],
        "metadatas": [[{k:v for k,v in res.payload.items() if k not in ["document", "original_id"]} for res in results]],
        "distances": [[1.0 - res.score for res in results]],
    }


def query_collection_sparse(
    collection_name: str,
    query: str,
    top_k: int = 5,
) -> dict:
    """
    Query the local TF-IDF sparse index.
    """
    sparse_index_path = os.path.join(PERSIST_DIR, f"{collection_name}_sparse.joblib")
    if not os.path.exists(sparse_index_path):
        print(f"  [VectorStore] Sparse index for '{collection_name}' not found.")
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        
    data = joblib.load(sparse_index_path)
    vectorizer = data['vectorizer']
    tfidf_matrix = data['matrix']
    docs = data['docs']
    metas = data['metas']
    
    query_vec = vectorizer.transform([query])
    cos_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Get top_k indices
    top_indices = cos_similarities.argsort()[-top_k:][::-1]
    
    selected_docs = [docs[i] for i in top_indices if cos_similarities[i] > 0]
    selected_metas = [metas[i] for i in top_indices if cos_similarities[i] > 0]
    selected_dists = [1.0 - cos_similarities[i] for i in top_indices if cos_similarities[i] > 0]
    
    return {
        "documents": [selected_docs],
        "metadatas": [selected_metas],
        "distances": [selected_dists],
    }


def reciprocal_rank_fusion(dense_results: list, sparse_results: list, k: int = 60) -> list:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion (RRF).
    Returns a sorted list of unique (doc, metadata) tuples.
    """
    rrf_scores = {}
    doc_metadata_map = {}
    
    # Process dense results
    for rank, (doc, meta) in enumerate(dense_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0
            doc_metadata_map[doc] = meta
        rrf_scores[doc] += 1.0 / (k + rank + 1)
        
    # Process sparse results
    for rank, (doc, meta) in enumerate(sparse_results):
        if doc not in rrf_scores:
            rrf_scores[doc] = 0
            doc_metadata_map[doc] = meta
        rrf_scores[doc] += 1.0 / (k + rank + 1)
        
    # Sort by RRF score descending
    sorted_docs = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)
    return [(doc, doc_metadata_map[doc]) for doc in sorted_docs]


def query_collection_diverse(
    collection_name: str,
    query: str,
    top_k: int = 5,
    fetch_k: int = None,
    lambda_mult: float = 0.5,
    where: dict = None,
) -> dict:
    """
    Hybrid Retrieval (Dense + Sparse) with Cross-Encoder Re-ranking and MMR Diversity.
    """
    if fetch_k is None:
        fetch_k = min(top_k * 10, 100)

    # 1. Broad Fetch using Dense Vector Search
    raw_dense = query_collection(collection_name, query, top_k=fetch_k, where=where)
    dense_docs = raw_dense["documents"][0]
    dense_metas = raw_dense["metadatas"][0]
    dense_results = list(zip(dense_docs, dense_metas))
    
    # 2. Broad Fetch using Sparse Vector Search (TF-IDF)
    raw_sparse = query_collection_sparse(collection_name, query, top_k=fetch_k)
    sparse_docs = raw_sparse["documents"][0]
    sparse_metas = raw_sparse["metadatas"][0]
    sparse_results = list(zip(sparse_docs, sparse_metas))

    if not dense_docs and not sparse_docs:
        return {"documents": [[]], "metadatas": [[]], "distances": [[]]}

    # 3. Hybrid Fusion via Reciprocal Rank Fusion (RRF)
    hybrid_candidates = reciprocal_rank_fusion(dense_results, sparse_results)
    cand_docs = [item[0] for item in hybrid_candidates]
    cand_metas = [item[1] for item in hybrid_candidates]

    # 4. Cross-Encoder Re-ranking
    model = get_cross_encoder()
    pairs = [[query, doc] for doc in cand_docs]
    scores = model.predict(pairs)
    
    # Normalize Cross-Encoder scores to [0, 1] range for MMR
    if len(scores) > 0:
        min_score = np.min(scores)
        max_score = np.max(scores)
        if max_score > min_score:
            norm_scores = (scores - min_score) / (max_score - min_score)
        else:
            norm_scores = np.ones_like(scores)
    else:
        norm_scores = scores

    # 5. MMR (Maximal Marginal Relevance) Diversity
    # We need dense embeddings of the candidates to compute diversity
    cand_embeddings = np.array(get_embeddings_batch(cand_docs))
    
    selected_indices = []
    unselected_indices = list(range(len(cand_docs)))
    
    # Initialize similarity matrix
    sim_matrix = cosine_similarity(cand_embeddings)
    
    while len(selected_indices) < top_k and unselected_indices:
        if not selected_indices:
            # First document is the one with highest relevance (normalized score)
            best_idx = max(unselected_indices, key=lambda i: norm_scores[i])
        else:
            # Calculate MMR score for remaining documents
            best_mmr = -float('inf')
            best_idx = -1
            
            for i in unselected_indices:
                # Similarity to already selected docs
                sim_to_selected = max([sim_matrix[i][j] for j in selected_indices])
                # MMR Equation
                mmr_score = lambda_mult * norm_scores[i] - (1 - lambda_mult) * sim_to_selected
                
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_idx = i
                    
        selected_indices.append(best_idx)
        unselected_indices.remove(best_idx)
    
    selected_docs = [cand_docs[i] for i in selected_indices]
    selected_metas = [cand_metas[i] for i in selected_indices]
    
    # For distances, we can return the inverted normalized relevance scores
    selected_dists = [1.0 - norm_scores[i] for i in selected_indices] 

    # Package result in standard ChromaDB format
    return {
        "documents": [selected_docs],
        "metadatas": [selected_metas],
        "distances": [selected_dists],
    }


def delete_collection(collection_name: str) -> None:
    try:
        if _client.collection_exists(collection_name=collection_name):
            _client.delete_collection(collection_name=collection_name)
            print(f"  [VectorStore] Deleted collection '{collection_name}'")
            
        sparse_index_path = os.path.join(PERSIST_DIR, f"{collection_name}_sparse.joblib")
        if os.path.exists(sparse_index_path):
            os.remove(sparse_index_path)
            print(f"  [VectorStore] Deleted sparse index '{collection_name}_sparse.joblib'")
    except Exception:
        pass


def get_collection_count(collection_name: str) -> int:
    """Return number of documents in a collection, or 0 if it doesn't exist."""
    try:
        if _client.collection_exists(collection_name=collection_name):
            return _client.get_collection(collection_name=collection_name).points_count
        return 0
    except Exception:
        return 0

