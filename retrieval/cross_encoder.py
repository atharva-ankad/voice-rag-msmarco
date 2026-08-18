import torch
import logging
from typing import List, Dict, Any
from sentence_transformers import CrossEncoder

logger = logging.getLogger(__name__)

class CrossEncoderReRanker:
    """
    Phase 9.5: Cross-Encoder Re-ranking layer.
    Computes full self-attention between the query and retrieved candidate chunks.
    """
    def __init__(
        self, 
        model_name: str = "BAAI/bge-reranker-v2-m3", 
        max_length: int = 512, 
        device: str = None
    ):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Brutally enforce FP16 on CUDA. Standard FP32 will OOM on mobile GPUs or T4s 
        # when processing 100 candidates with max_length=512.
        model_kwargs = {"torch_dtype": torch.float16} if "cuda" in self.device else {}
        
        logger.info(f"Loading CrossEncoder: {model_name} on {self.device} in {model_kwargs.get('torch_dtype', 'FP32')}")
        
        self.reranker = CrossEncoder(
            model_name,
            max_length=max_length,
            device=self.device,
            model_kwargs=model_kwargs
        )

    def rerank(
        self, 
        query: str, 
        candidates: List[Dict[str, Any]], 
        top_k: int = 5, 
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Re-ranks candidate chunks. 
        `candidates` must contain at least 'chunk_id' and 'text'.
        """
        if not candidates:
            return []

        # The cross-encoder expects input as [[query, text1], [query, text2], ...]
        sentence_pairs = [[query, doc["text"]] for doc in candidates]

        try:
            # Predict scores. batch_size limits concurrent VRAM consumption.
            scores = self.reranker.predict(sentence_pairs, batch_size=batch_size)
        except torch.cuda.OutOfMemoryError as e:
            logger.error("CUDA OOM during reranking. Halving candidate pool as fallback constraint.")
            # Immediate fallback: Clear cache and retry with top 50 candidates
            torch.cuda.empty_cache()
            return self.rerank(query, candidates[:50], top_k, batch_size)

        # Inject scores back into the candidate dictionaries
        for idx, score in enumerate(scores):
            # Convert float32 numpy outputs to native Python floats for JSON/API compatibility
            candidates[idx]["cross_encoder_score"] = float(score)

        # Sort descending by the new cross-encoder score
        reranked_candidates = sorted(
            candidates, 
            key=lambda x: x["cross_encoder_score"], 
            reverse=True
        )

        return reranked_candidates[:top_k]