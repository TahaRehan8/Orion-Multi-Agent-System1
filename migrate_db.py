import os
import chromadb
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHROMA_PERSIST_DIR = os.path.join(PROJECT_ROOT, "chroma_db")
QDRANT_PERSIST_DIR = os.path.join(PROJECT_ROOT, "qdrant_db")

def migrate():
    print("Initializing clients...")
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    qdrant_client = QdrantClient(path=QDRANT_PERSIST_DIR)
    
    collections = ["finance_data", "hr_data", "scheduler_data"]
    
    for collection_name in collections:
        print(f"\nMigrating collection: {collection_name}")
        
        chroma_col = chroma_client.get_collection(collection_name)
        data = chroma_col.get(include=["embeddings", "documents", "metadatas"])
        
        ids = data["ids"]
        embeddings = data["embeddings"]
        documents = data["documents"]
        metadatas = data["metadatas"]
        
        if not ids:
            print(f"Collection {collection_name} is empty. Skipping.")
            continue
            
        vector_size = len(embeddings[0])
        print(f"Found {len(ids)} documents. Vector dimension: {vector_size}")
        
        # Recreate Qdrant collection
        if qdrant_client.collection_exists(collection_name=collection_name):
            qdrant_client.delete_collection(collection_name=collection_name)
            
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        
        # Batch insert into Qdrant
        points = []
        for i in range(len(ids)):
            # Qdrant requires UUIDs or integers for IDs. If Chroma used string IDs, we hash them or just use index.
            # But wait, Qdrant allows string UUIDs, but Chroma IDs might not be UUIDs.
            # Let's generate standard string UUIDs based on the string.
            import uuid
            qdrant_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, ids[i]))
            
            payload = metadatas[i] if metadatas[i] else {}
            payload["document"] = documents[i]
            payload["original_id"] = ids[i]
            
            points.append(
                PointStruct(
                    id=qdrant_id,
                    vector=embeddings[i],
                    payload=payload
                )
            )
            
        # Insert in batches
        BATCH_SIZE = 256
        for start in range(0, len(points), BATCH_SIZE):
            end = start + BATCH_SIZE
            qdrant_client.upsert(
                collection_name=collection_name,
                points=points[start:end]
            )
            print(f"  Inserted {min(end, len(points))}/{len(points)}")
            
    print("\nMigration complete! Data is now safely in qdrant_db/")

if __name__ == "__main__":
    migrate()
