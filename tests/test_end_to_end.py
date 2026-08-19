import logging
from retrieval.hybrid_search import HybridRetriever
from generation.generator import RAGGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_end_to_end_pipeline():
    print("\n" + "="*50)
    print("INITIALIZING FULL END-TO-END RAG PIPELINE")
    print("="*50)
    
    retriever = HybridRetriever()
    generator = RAGGenerator()
    
    query = "What causes high blood pressure?"
    print(f"\nProcessing Query: '{query}'")
    
    # 1. Hybrid Retrieval + Cross-Encoder Reranking (Phase 9 + 9.5)
    top_reranked = retriever.hybrid_search(
        query=query, 
        final_top_k=3,          # Keep top 3 for generation context
        candidate_pool_size=100
    )
    
    # 2. Prompt Construction (Phase 11)
    final_prompt = generator.generate_prompt(query, top_reranked)
    
    print("\n" + "-"*50)
    print("GENERATED PROMPT PAYLOAD FOR LLM:")
    print("-" * 50)
    print(final_prompt)
    print("-" * 50)

if __name__ == "__main__":
    run_end_to_end_pipeline()