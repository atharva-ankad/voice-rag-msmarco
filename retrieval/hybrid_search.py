import os
import pickle
import torch
import sqlite3
from typing import List, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client.http import models
from FlagEmbedding import BGEM3FlagModel
from pathlib import Path
from dotenv import load_dotenv

from retrieval.cross_encoder import CrossEncoderReRanker

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Configuration ---
# Dynamically anchor to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "rag_sidecar.db")

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "msmarco_multilingual_bge"
BM25_INDEX_PATH = os.path.join(BASE_DIR, "cache", "bm25_semantic_300.pkl")

# RRF Smoothing Constant
K = 60

class HybridRetriever:
    def __init__(self, strategy_filter: str = "semantic_300", db_path: str = DB_PATH):
        self.strategy = strategy_filter
        
        print(f"Connecting to SQLite sidecar at {db_path}...")
        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database not found at {db_path}")
            
        # 1. Convert the file path to an absolute URI for read-only mode
        db_uri = f"file:{os.path.abspath(db_path)}?mode=ro"
        
        # 2. Connect using the URI flag
        self.db_conn = sqlite3.connect(db_uri, uri=True, check_same_thread=False)

        # 3. Inject extreme read-only optimizations
        self.db_conn.execute("PRAGMA mmap_size = 268435456;") # Map 256MB directly into RAM
        self.db_conn.execute("PRAGMA journal_mode = OFF;")    # Disable disk write logs completely
        self.db_conn.execute("PRAGMA query_only = ON;")       # Hard lock to prevent accidental writes

        print("Loading Qdrant Client...")
        self.qdrant = QdrantClient(
            url=QDRANT_URL, 
            api_key=QDRANT_API_KEY,
            check_compatibility=False 
        )
        
        print("Loading BAAI/bge-m3 (FP16)...")
        self.model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
        
        print(f"Loading BM25 Index from {BM25_INDEX_PATH}...")
        self.bm25, self.bm25_chunk_ids = self._load_bm25()

        print("Loading Cross-Encoder (BAAI/bge-reranker-v2-m3)...")
        self.reranker = CrossEncoderReRanker(model_name="BAAI/bge-reranker-v2-m3")

    def __del__(self):
        """Ensure SQLite connection closes cleanly."""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()

    def _load_bm25(self):
        if not os.path.exists(BM25_INDEX_PATH):
            raise FileNotFoundError(f"BM25 index not found at {BM25_INDEX_PATH}")
            
        with open(BM25_INDEX_PATH, 'rb') as f:
            data = pickle.load(f)
            
        # Match the exact dictionary keys used in retrieval/bm25_index.py
        # Example: return data['bm25'], data['chunk_ids']
        return data['bm25'], data['chunk_ids']

    def dense_search(self, query: str, top_k: int = 50) -> Dict[str, float]:
        """Returns a dict mapping chunk_id to its Dense Rank (1-indexed)."""
        query_vector = self.model.encode([query], return_dense=True)['dense_vecs'][0].tolist()
        
        # Updated to use the new query_points API
        response = self.qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            query_filter=models.Filter(
                must=[models.FieldCondition(key="strategy", match=models.MatchValue(value=self.strategy))]
            ),
            limit=top_k,
            with_payload=["original_chunk_id", "text"]
        )
        
        # Map original_chunk_id to rank (1, 2, 3...)
        # Note: query_points returns a response object; the list is accessed via .points
        return {hit.payload["original_chunk_id"]: (rank + 1, hit.payload["text"]) 
                for rank, hit in enumerate(response.points)}

    def lexical_search(self, query: str, top_k: int = 50) -> Dict[str, int]:
        """Returns a dict mapping chunk_id to its BM25 Rank (1-indexed)."""
        # Tokenization must match Phase 8 logic exactly
        tokenized_query = query.lower().split() 
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort and get top K
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        return {self.bm25_chunk_ids[i]: rank + 1 for rank, i in enumerate(top_indices)}


    def hybrid_search(self, query: str, final_top_k: int = 5, candidate_pool_size: int = 100) -> List[Dict[str, Any]]:
        """Executes Reciprocal Rank Fusion, resolves payloads, and applies Cross-Encoder Re-ranking."""
        print(f"\nExecuting Hybrid Search for: '{query}'")
        
        # 1. Expand initial retrieval depth to ensure a rich pool for the cross-encoder
        dense_ranks = self.dense_search(query, top_k=candidate_pool_size)
        lexical_ranks = self.lexical_search(query, top_k=candidate_pool_size)
        
        all_chunk_ids = set(dense_ranks.keys()).union(set(lexical_ranks.keys()))
        
        rrf_scores = []
        missing_text_ids = []
        K = 60 # Standard RRF constant
        
        # 2. Compute RRF Scores
        for chunk_id in all_chunk_ids:
            dense_score = 0.0
            text = None
            if chunk_id in dense_ranks:
                dense_score = 1.0 / (K + dense_ranks[chunk_id][0])
                text = dense_ranks[chunk_id][1] # Extracted from Qdrant payload
                
            lexical_score = 0.0
            if chunk_id in lexical_ranks:
                lexical_score = 1.0 / (K + lexical_ranks[chunk_id])
                
            # Track chunks that BM25 found but Qdrant missed
            if text is None:
                missing_text_ids.append(chunk_id)
                
            total_score = dense_score + lexical_score
            rrf_scores.append({"chunk_id": chunk_id, "score": total_score, "text": text})
            
        # 3. Bulk Data Resolution (The Fix)
        # Fetch all missing text payloads from SQLite in a single transaction
        if missing_text_ids:
            text_mapping = self._fetch_text_from_sqlite_bulk(missing_text_ids)
            for item in rrf_scores:
                if item["text"] is None:
                    item["text"] = text_mapping.get(item["chunk_id"], "")
                    
        # Sort by RRF score and truncate to the cross-encoder candidate pool limit
        rrf_scores.sort(key=lambda x: x["score"], reverse=True)
        top_candidates = rrf_scores[:candidate_pool_size]
        
        # 4. Phase 9.5: Cross-Encoder Reranking
        print(f"Passing {len(top_candidates)} candidates to BAAI/bge-reranker-v2-m3...")
        final_results = self.reranker.rerank(
            query=query, 
            candidates=top_candidates, 
            top_k=final_top_k
        )
        
        return final_results
    
    def _fetch_text_from_sqlite_bulk(self, chunk_ids: List[str]) -> Dict[str, str]:
        """
        Retrieves missing chunk texts in a single parameterized SQL query.
        Prevents N+1 query bottlenecks during retrieval.
        """
        if not chunk_ids:
            return {}
            
        # Create parameter placeholders (?, ?, ?) dynamically based on list size
        placeholders = ",".join(["?"] * len(chunk_ids))
        query = f"SELECT chunk_id, text FROM chunks WHERE chunk_id IN ({placeholders})"
        
        cursor = self.db_conn.cursor()
        cursor.execute(query, chunk_ids)
        
        # Return a mapping of chunk_id -> text
        return {row[0]: row[1] for row in cursor.fetchall()}

if __name__ == "__main__":
    retriever = HybridRetriever(strategy_filter="semantic_300")
    results = retriever.hybrid_search("What causes high blood pressure?")
    
    for i, res in enumerate(results, 1):
        print(f"\n[{i}] RRF Score: {res['score']:.6f} | ID: {res['chunk_id']}")
        print(f"Text: {res['text'][:200]}...")