import time
import torch
import logging
from retrieval.hybrid_search import HybridRetriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_sanity_check():
    # 1. Initialize Pipeline
    print("\n" + "="*50)
    print("INITIALIZING HYBRID RETRIEVER & CROSS-ENCODER")
    print("="*50)
    
    start_init = time.perf_counter()
    retriever = HybridRetriever()
    init_time = time.perf_counter() - start_init
    print(f"Initialization completed in {init_time:.2f}s")
    
    # 2. Test Query (Choose one that tests causal/antonym ambiguity)
    test_queries = [
        "What causes high blood pressure?",
        "उच्च रक्तचाप के क्या कारण हैं?"  # Multilingual sanity check
    ]
    
    for query in test_queries:
        print("\n" + "-"*50)
        print(f"Testing Query: '{query}'")
        print("-" * 50)
        
        # Track memory before inference
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            vram_start = torch.cuda.memory_allocated() / (1024 ** 2)
            print(f"VRAM Allocated before search: {vram_start:.2f} MB")
            
        start_time = time.perf_counter()
        
        # Run retrieval with 100 candidate depth down to top 5
        results = retriever.hybrid_search(
            query=query, 
            final_top_k=5, 
            candidate_pool_size=100
        )
        
        elapsed_time = time.perf_counter() - start_time
        
        # 3. Inspect Results & Diagnostics
        print(f"\nCompleted in {elapsed_time:.2f}s")
        if torch.cuda.is_available():
            peak_vram = torch.cuda.max_memory_allocated() / (1024 ** 2)
            print(f"Peak VRAM during search + rerank: {peak_vram:.2f} MB")
            
        print(f"\nTop {len(results)} Reranked Results:")
        for idx, item in enumerate(results, start=1):
            chunk_id = item.get("chunk_id")
            score = item.get("cross_encoder_score")
            text = item.get("text", "")
            
            # Assertions to ensure data integrity
            assert text is not None and text != "", f"FAILED: Empty text for chunk {chunk_id}"
            assert "Text lookup needed" not in text, f"FAILED: Unresolved placeholder text for chunk {chunk_id}"
            
            snippet = text[:120].replace("\n", " ")
            print(f"[{idx}] Score: {score:+.4f} | Chunk ID: {chunk_id} | Snippet: {snippet}...")

    print("\n" + "="*50)
    print("ALL SANITY CHECKS PASSED SUCCESSFULLY")
    print("="*50)

if __name__ == "__main__":
    run_sanity_check()