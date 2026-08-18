#Colab Script
'''import sqlite3
import os
import time
import uuid
import torch
from typing import List
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from FlagEmbedding import BGEM3FlagModel
from tqdm.notebook import tqdm
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Colab Configuration ---
# Hardcoded to the Google Drive path we created in Step 1
SQLITE_DB_PATH = "/content/drive/MyDrive/RAG_Project/rag_sidecar.db"

# TODO: Paste your Qdrant credentials here
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "msmarco_multilingual_bge"

# T4 GPU SCALING:
BATCH_SIZE = 256  # Increased from 16 to saturate 16GB VRAM
MAX_RETRIES = 5   # Increased retries for network resilience
API_THROTTLE = 0.0 # Seconds to sleep between uploads to prevent 429 Rate Limits

# --- Hardware Verification ---
assert torch.cuda.is_available(), "CRITICAL: Colab runtime is not connected to a GPU. Change runtime type to T4."
print(f"Device: {torch.cuda.get_device_name(0)}")
print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")

# --- Model Initialization ---
print("\nLoading BAAI/bge-m3 into T4 VRAM (FP16)...")
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

def get_embeddings(texts: List[str]) -> List[List[float]]:
    output = model.encode(
        texts,
        batch_size=len(texts),
        max_length=512,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    return output['dense_vecs'].tolist()

def init_qdrant_collection(client: QdrantClient):
    if not client.collection_exists(COLLECTION_NAME):
        print(f"Creating Qdrant collection '{COLLECTION_NAME}' (dim=1024)...")
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists. Resuming.")

def get_total_chunks(db_path: str) -> int:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM chunks")
    total = cursor.fetchone()[0]
    conn.close()
    return total

def fetch_chunks_in_batches(db_path: str, batch_size: int):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row 
    cursor = conn.cursor()
    cursor.execute("SELECT chunk_id, passage_id, strategy, text, token_count FROM chunks")
    
    while True:
        results = cursor.fetchmany(batch_size)
        if not results:
            break
        yield [dict(row) for row in results]
    conn.close()

def main():
    if QDRANT_URL == "YOUR_QDRANT_URL":
        raise ValueError("You forgot to paste your Qdrant credentials.")

    qdrant = QdrantClient(
        url=QDRANT_URL, 
        api_key=QDRANT_API_KEY,
        timeout=60.0,
        prefer_grpc=True
    )
    init_qdrant_collection(qdrant)

    total_chunks = get_total_chunks(SQLITE_DB_PATH)
    print(f"\nStarting ingestion for {total_chunks} chunks...")
    
    # Use tqdm for a clean progress bar in Colab
    pbar = tqdm(total=total_chunks, desc="Embedding & Uploading")
    
    for batch in fetch_chunks_in_batches(SQLITE_DB_PATH, BATCH_SIZE):
        texts = [row["text"] for row in batch]
        
        # 1. GPU Inference
        try:
            vectors = get_embeddings(texts)
        except Exception as e:
            print(f"\nCUDA Error: {e}")
            break

        # 2. Map Payloads
        points = []
        for i, row in enumerate(batch):
            safe_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, row["chunk_id"]))
            points.append(
                PointStruct(
                    id=safe_id,
                    vector=vectors[i],
                    payload={
                        "original_chunk_id": row["chunk_id"],
                        "passage_id": row["passage_id"],
                        "strategy": row["strategy"],
                        "text": row["text"],
                        "token_count": row["token_count"]
                    }
                )
            )
        
        # 3. Network Upload (Asynchronous Server Acknowledgment)
        uploaded = False
        for attempt in range(1, MAX_RETRIES + 1):
          try:
              qdrant.upsert(
                collection_name=COLLECTION_NAME, 
                points=points,
                wait=False  # Fire-and-forget into Qdrant's ingestion queue
              )
              uploaded = True
              break
          except Exception as e:
            wait_time = 2 ** attempt
            print(f"\nUpload failed. Retrying in {wait_time}s... Error: {e}")
            time.sleep(wait_time)
        
        if not uploaded:
            print(f"\nFATAL: Failed to upload batch starting with chunk_id: {batch[0]['chunk_id']}")
            break

        # 4. Progress Update & Artificial Throttle
        pbar.update(len(batch))
        time.sleep(API_THROTTLE) # Prevents the T4 from DDoS-ing Qdrant

    pbar.close()
    print("\nPipeline Complete.")

if __name__ == "__main__":
    main()'''