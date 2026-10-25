"""
Shared embedding function so ingest and retrieve always use the same model.
"""

import os
from pathlib import Path
from chromadb.utils import embedding_functions

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = str(ROOT / "chroma_db")
COLLECTION = "pdf_documents"

# Served by the local Ollama server, which manages where the model files live
EMBEDDING_MODEL_NAME = "nomic-embed-text"
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")  # Docker sets http://ollama:11434

def get_embedding_function():
    return embedding_functions.OllamaEmbeddingFunction(url=OLLAMA_URL, model_name=EMBEDDING_MODEL_NAME)
