import os
import logging
from sarvamai import SarvamAI
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

class RAGGenerator:
    """
    Phase 11: LLM Synthesis Layer using the official SarvamAI SDK.
    """
    def __init__(self, system_prompt: str = None):
        self.system_prompt = system_prompt or (
            "You are a precise multilingual AI assistant for the AI4Bharat MSMARCO dataset. "
            "Answer the user's query strictly using ONLY the provided context chunks. "
            "If the context does not contain the answer, state clearly that you do not know. "
            "Do not hallucinate. Match the language of the query."
        )
        
        # Initialize SarvamAI client
        api_key = os.getenv("SARVAM_API_KEY")
        if not api_key:
            raise ValueError("SARVAM_API_KEY environment variable is missing. Set it before running.")
            
        self.client = SarvamAI(api_subscription_key=api_key)

    def build_context_window(self, reranked_candidates: List[Dict[str, Any]]) -> str:
        """Formats top chunks into a clean context block."""
        context_blocks = []
        for idx, item in enumerate(reranked_candidates, start=1):
            text = item.get("text", "")
            chunk_id = item.get("chunk_id", "unknown")
            context_blocks.append(f"[Source {idx} | ID: {chunk_id}]\n{text}")
        return "\n\n".join(context_blocks)

    def generate_response(self, query: str, reranked_candidates: List[Dict[str, Any]]) -> str:
            """Calls the Sarvam-105B API using the official SDK."""
            context_window = self.build_context_window(reranked_candidates)
            
            user_content = f"### Retrieved Context Passages\n{context_window}\n\n### User Query\n{query}\n\n### Answer:"
            
            try:
                response = self.client.chat.completions(
                    model="sarvam-105b",
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": user_content}
                    ],
                    temperature=0.1,  # Keep low for strict RAG grounding
                    top_p=1,
                    max_tokens=512,
                    reasoning_effort=None  # <--- explicitly disable "Thinking Mode"
                )
                
                # Fallback to prevent NoneType errors if the model hits another cutoff
                content = response.choices[0].message.content
                return content if content else "Error: Model returned an empty string."
                
            except Exception as e:
                logger.error(f"Generation failed: {str(e)}")
                raise