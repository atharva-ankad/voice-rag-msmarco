import os
import tempfile
import logging
import httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Initialize app ONLY ONCE
app = FastAPI(title="Voice-Enabled RAG Pipeline")

from interfaces.voice_handler import VoiceHandler
from retrieval.hybrid_search import HybridRetriever
from retrieval.cross_encoder import CrossEncoderReRanker
from generation.generator import RAGGenerator
from generation.guardrails import GuardrailManager

# Allows Vercel frontend to talk to the Modal backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Configure logging for the harness
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Structured Output Model
class RAGResponse(BaseModel):
    query: str
    response: str
    safe: bool
    error: str | None = None

# Initialize modules once at startup to keep models loaded in memory
voice_handler = VoiceHandler()
retriever = HybridRetriever()
reranker = CrossEncoderReRanker()
generator = RAGGenerator()
guardrails = GuardrailManager()

@app.post("/chat/audio", response_model=RAGResponse)
async def chat_audio(audio: UploadFile = File(...)):
    """
    Structured orchestration endpoint. 
    Accepts an audio file, transcribes/translates it, and runs it through the secure RAG pipeline.
    """
    file_extension = ".wav" 
    if audio.filename and "." in audio.filename:
        file_extension = f".{audio.filename.split('.')[-1]}"
        
    with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_audio:
        temp_audio.write(await audio.read())
        temp_audio_path = temp_audio.name
        
    try:
        # Voice-to-Text
        logger.info("Starting Speech-to-Text translation...")
        query = voice_handler.transcribe_and_translate(temp_audio_path)
        logger.info(f"Transcribed Query: {query}")
        
        # Guardrail 1: Input Safety (Properly unpacked tuple)
        is_safe, safety_reason = guardrails.is_input_safe(query)
        if not is_safe:
            logger.warning(f"Input rejected by safety guardrails: {safety_reason}")
            return RAGResponse(
                query=query, 
                response="Input rejected by safety guardrails.", 
                safe=False, 
                error="unsafe_input"
            )
            
        # Retrieval & Re-ranking 
        logger.info("Executing Hybrid Search and Cross-Encoder Reranking...")
        candidates = retriever.hybrid_search(query)
        top_candidates = reranker.rerank(query, candidates)
        
        # Guardrail 2: Context Relevance
        top_score = top_candidates[0].get("score", top_candidates[0].get("rerank_score", -1.0)) if top_candidates else -1.0
        if not guardrails.is_context_relevant(top_score):
            logger.warning("No relevant context found in vector DB.")
            return RAGResponse(
                query=query, 
                response="I cannot answer this based on the provided context.", 
                safe=True,
                error="context_irrelevant"
            )
             
        # Generation
        logger.info("Generating response...")
        answer, context_used = generator.generate_response(query, top_candidates)
        
        # Guardrail 3: Hallucination Check
        if not guardrails.is_output_grounded(context_used, answer):
            logger.warning("Response failed grounding verification.")
            return RAGResponse(
                query=query, 
                response="The generated response failed grounding verification.", 
                safe=False,
                error="hallucination_detected"
            )
            
        return RAGResponse(
            query=query,
            response=answer,
            safe=True
        )
        
    except httpx.ReadTimeout:
        logger.error("Generation timed out.")
        return RAGResponse(
            query=query if 'query' in locals() else "Unknown",
            response="The system timed out processing your request. Please try again.",
            safe=False,
            error="api_timeout"
        )
    except Exception as e:
        logger.error(f"Pipeline error: {str(e)}")
        # Structured error handling instead of raw 500 crashes
        return RAGResponse(
            query=query if 'query' in locals() else "Unknown",
            response="Internal processing error.",
            safe=False,
            error="internal_error"
        )
        
    finally:
        # Cleanup temp files
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)