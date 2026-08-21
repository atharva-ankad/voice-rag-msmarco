import modal
import os

# 1. Image Build & Dependency Management
def download_bge_models():
    """
    Downloads BAAI embedding and cross-encoder weights during the container build.
    This prevents the container from timing out when it wakes up from an idle state.
    """
    import torch
    from FlagEmbedding import BGEM3FlagModel
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    
    print("Caching BGE-M3 (FP16)...")
    BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    print("Caching Cross-Encoder...")
    AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    AutoModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3",
        torch_dtype=torch.float16
    )

# Define the container environment
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
    .run_function(download_bge_models)

    # Application source
    .add_local_file(
        "app.py",
        remote_path="/root/app.py",
        copy=True,
    )
    .add_local_dir(
        "chunking",
        remote_path="/root/chunking",
        copy=True,
    )
    .add_local_dir(
        "config",
        remote_path="/root/config",
        copy=True,
    )
    .add_local_dir(
        "generation",
        remote_path="/root/generation",
        copy=True,
    )
    .add_local_dir(
        "interfaces",
        remote_path="/root/interfaces",
        copy=True,
    )
    .add_local_dir(
        "retrieval",
        remote_path="/root/retrieval",
        copy=True,
    )

    # Runtime data
    .add_local_file(
        "rag_sidecar.db",
        remote_path="/root/rag_sidecar.db",
        copy=True,
    )
    .add_local_file(
        "cache/bm25_semantic_300.pkl",
        remote_path="/root/cache/bm25_semantic_300.pkl",
        copy=True,
    )
)

# 2. Define the Application
app = modal.App("hacker-house-goa-rag")

# 3. Mount the ASGI App
@app.function(
    image=image,
    gpu="T4",  # Required for FP16 FlagEmbedding performance
    secrets=[
        # You must create these secrets in the Modal dashboard beforehand
        modal.Secret.from_name("rag-keys")
    ]
)
@modal.asgi_app()
def fastapi_app():
    """
    Imports your existing FastAPI app instance from app.py.
    Modal will automatically mount the surrounding local .py files.
    """
    # Assuming your refactored code from earlier is saved as app.py
    from app import app as web_app
    return web_app