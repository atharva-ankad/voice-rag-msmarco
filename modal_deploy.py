import modal

# 1. Define the container environment (This replaces your Dockerfile)
# It pulls a base Linux image, installs your requirements, and copies your local files
rag_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements("requirements.txt")
    .pip_install("edge-tts") # Catching the extra dependency we added
    .add_local_dir("retrieval", remote_path="/root/retrieval")
    .add_local_dir("generation", remote_path="/root/generation")
    .add_local_dir("interfaces", remote_path="/root/interfaces")
    .add_local_file("rag_sidecar.db", remote_path="/root/rag_sidecar.db", copy=False)
    .add_local_dir("cache", remote_path="/root/cache")
)

# 2. Initialize the Modal App
app = modal.App("voice-rag-msmarco")

# 3. Mount the FastAPI app to a serverless GPU endpoint
@app.function(image=rag_image, gpu="T4")
@modal.asgi_app()
def fastapi_endpoint():
    # Import your existing FastAPI app instance from your app.py file
    from app import app as fastapi_app
    return fastapi_app