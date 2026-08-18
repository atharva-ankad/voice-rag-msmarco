import sqlite3
import os
import pickle
import re
from rank_bm25 import BM25Okapi

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'rag_sidecar.db')
INDEX_CACHE_DIR = os.path.join(os.path.dirname(__file__), '..', 'cache')

def tokenize_indic_latin(text: str) -> list[str]:
    """Tokenize English words and Indic unicode tokens."""
    return re.findall(r'\w+', text.lower(), re.UNICODE)

class BM25SearchEngine:
    def __init__(self, strategy: str = "semantic_300"):
        self.strategy = strategy
        self.chunk_ids: list[str] = []
        self.bm25: BM25Okapi | None = None

    def build_and_save(self, cache_file: str | None = None):
        os.makedirs(INDEX_CACHE_DIR, exist_ok=True)
        if cache_file is None:
            cache_file = os.path.join(INDEX_CACHE_DIR, f"bm25_{self.strategy}.pkl")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        print(f"Loading chunks for strategy: {self.strategy}...")
        cursor.execute("SELECT chunk_id, text FROM chunks WHERE strategy = ?", (self.strategy,))
        rows = cursor.fetchall()
        conn.close()

        print(f"Tokenizing {len(rows)} chunks for BM25...")
        self.chunk_ids = [row[0] for row in rows]
        tokenized_corpus = [tokenize_indic_latin(row[1]) for row in rows]

        print("Fitting BM25Okapi index...")
        self.bm25 = BM25Okapi(tokenized_corpus)

        with open(cache_file, "wb") as f:
            pickle.dump({"chunk_ids": self.chunk_ids, "bm25": self.bm25}, f)
        print(f"BM25 index saved to {cache_file}")

    def load(self, cache_file: str | None = None):
        if cache_file is None:
            cache_file = os.path.join(INDEX_CACHE_DIR, f"bm25_{self.strategy}.pkl")
        
        if not os.path.exists(cache_file):
            print(f"Cache file {cache_file} not found. Building now...")
            self.build_and_save(cache_file)
            return

        with open(cache_file, "rb") as f:
            data = pickle.load(f)
            self.chunk_ids = data["chunk_ids"]
            self.bm25 = data["bm25"]
        print(f"BM25 index for '{self.strategy}' loaded ({len(self.chunk_ids)} docs).")

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        if self.bm25 is None:
            raise ValueError("BM25 index is not loaded. Call .load() or .build_and_save() first.")
        
        tokenized_query = tokenize_indic_latin(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [(self.chunk_ids[i], float(scores[i])) for i in top_indices if scores[i] > 0]

if __name__ == "__main__":
    # Test building and loading index for semantic_300 strategy
    engine = BM25SearchEngine(strategy="semantic_300")
    engine.build_and_save()
    
    # Test sample search
    sample_query = "What is the capital city?"
    results = engine.search(sample_query, top_k=5)
    print("\nSample BM25 Search Results:")
    for chunk_id, score in results:
        print(f"Chunk ID: {chunk_id} | Score: {score:.4f}")