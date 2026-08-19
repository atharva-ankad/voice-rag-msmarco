import logging
from retrieval.hybrid_search import HybridRetriever
from generation.generator import RAGGenerator

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def test_pipeline_handoff():
    print("=" * 60)
    print("INITIALIZING RETRIEVER & GENERATOR")
    print("=" * 60)
    retriever = HybridRetriever()
    generator = RAGGenerator()
    
    query = "What is the main risk factor for stroke?, Hindi answer."
    print(f"\nTarget Query: '{query}'")
    
    # 1. Retrieve & Rerank
    top_candidates = retriever.hybrid_search(
        query=query, 
        final_top_k=3, 
        candidate_pool_size=100
    )
    
    # 2. Inspect Retrieved Chunks
    print("\n" + "=" * 60)
    print(f"TOP {len(top_candidates)} CHUNKS PASSED TO LLM CONTEXT:")
    print("=" * 60)
    
    for idx, doc in enumerate(top_candidates, start=1):
        chunk_id = doc.get("chunk_id", "N/A")
        score = doc.get("cross_encoder_score", 0.0)
        text = doc.get("text", "")
        
        print(f"\n--- [Chunk #{idx}] ---")
        print(f"Chunk ID:             {chunk_id}")
        print(f"Cross-Encoder Score:  {score:+.4f}")
        print(f"Text Length:          {len(text)} chars")
        print("Raw Content:")
        print(f'"{text}"')
    
    # 3. Inspect Full Constructed Prompt
    print("\n" + "=" * 60)
    print("EXACT PROMPT SENT TO SARVAM-105B:")
    print("=" * 60)
    context_block = generator.build_context_window(top_candidates)
    prompt_payload = f"### System Instructions\n{generator.system_prompt}\n\n### Retrieved Context Passages\n{context_block}\n\n### User Query\n{query}\n\n### Answer:"
    print(prompt_payload)
    
    # 4. Generate Answer
    print("\n" + "=" * 60)
    print("CALLING SARVAM-105B GENERATION...")
    print("=" * 60)
    final_answer = generator.generate_response(query, top_candidates)
    
    print("\n" + "=" * 60)
    print("FINAL RAG OUTPUT:")
    print("=" * 60)
    print(final_answer)
    print("=" * 60)

if __name__ == "__main__":
    test_pipeline_handoff()