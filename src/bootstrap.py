"""Container startup: wait for Ollama, pull the models the app needs, and build the index if it is missing."""
import time
import chromadb
import ollama
import ingest
from embeddings import CHROMA_PATH, COLLECTION, EMBEDDING_MODEL_NAME, OLLAMA_URL
from llm import load_config

client = ollama.Client(host=OLLAMA_URL)
for _ in range(60):
    try:
        installed = {m.model for m in client.list().models}
        break
    except Exception:
        print(f"Waiting for Ollama at {OLLAMA_URL}...", flush=True)
        time.sleep(2)
else:
    raise SystemExit(f"Ollama not reachable at {OLLAMA_URL}")

for model in [EMBEDDING_MODEL_NAME, load_config()["model"]]:
    if model not in installed and f"{model}:latest" not in installed:
        print(f"Pulling {model} (first start only)...", flush=True)
        client.pull(model)

if COLLECTION not in [c.name for c in chromadb.PersistentClient(path=CHROMA_PATH).list_collections()]:
    print("No index found, ingesting data/ ...", flush=True)
    ingest.main()
print("Ready.", flush=True)
