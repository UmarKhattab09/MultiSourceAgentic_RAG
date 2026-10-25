# src/retrieve.py
import re
from functools import lru_cache
from pathlib import Path
import chromadb
from langsmith import traceable
from rank_bm25 import BM25Okapi
from embeddings import get_embedding_function, CHROMA_PATH, COLLECTION
from llm import query_ollama, load_config

CANDIDATES = 20  # how many results each retriever feeds into fusion / the reranker

def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())

@lru_cache
def get_collection():
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=COLLECTION, embedding_function=get_embedding_function())

@lru_cache
def get_bm25():
    """BM25 index over every chunk in Chroma, so both retrievers see the same corpus."""
    data = get_collection().get(include=["documents", "metadatas"])
    return BM25Okapi([tokenize(d) for d in data["documents"]]), data

@lru_cache
def get_reranker():
    from sentence_transformers import CrossEncoder
    return CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def dense_search(query: str, k: int) -> list[tuple[str, dict]]:
    r = get_collection().query(query_texts=[query], n_results=k)
    return list(zip(r["documents"][0], r["metadatas"][0]))

def bm25_search(query: str, k: int) -> list[tuple[str, dict]]:
    bm25, data = get_bm25()
    scores = bm25.get_scores(tokenize(query))
    best = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
    return [(data["documents"][i], data["metadatas"][i]) for i in best]

def rrf(result_lists: list[list[tuple[str, dict]]], k: int = 60) -> list[tuple[str, dict]]:
    """Reciprocal Rank Fusion: score = sum of 1/(k + rank) across lists."""
    scores, items = {}, {}
    for results in result_lists:
        for rank, (doc, meta) in enumerate(results):
            scores[doc] = scores.get(doc, 0) + 1 / (k + rank + 1)
            items[doc] = (doc, meta)
    return [items[d] for d in sorted(scores, key=lambda d: -scores[d])]

@traceable(name="retrieve")  # no-op unless LangSmith tracing is on
def search(query: str, mode: str | None = None, top_k: int = 5) -> list[tuple[str, dict]]:
    """Return top_k (text, metadata) chunks. mode: dense | hybrid | hybrid_rerank (default from config)."""
    mode = mode or load_config()["retrieval_mode"]
    if mode == "dense":
        return dense_search(query, top_k)
    fused = rrf([dense_search(query, CANDIDATES), bm25_search(query, CANDIDATES)])[:CANDIDATES]
    if mode == "hybrid":
        return fused[:top_k]
    if mode == "hybrid_rerank":
        scores = get_reranker().predict([(query, doc) for doc, _ in fused])
        ranked = sorted(zip(scores, fused), key=lambda x: -x[0])
        return [item for _, item in ranked[:top_k]]
    raise ValueError(f"Unknown retrieval_mode: {mode}")

def format_context(chunks: list[tuple[str, dict]]) -> str:
    blocks = []
    for doc, meta in chunks:
        page = f", page {meta['page_number']}" if "page_number" in meta else ""
        blocks.append(f"[Source: {Path(meta['source']).name}{page}]\n{doc}")
    return "\n---\n".join(blocks)

def retrieve_context(query: str, top_k: int = 5, mode: str | None = None) -> str:
    """Retrieve chunks and return them as one context string with source labels."""
    return format_context(search(query, mode, top_k))

def query_rag(question: str):
    """Retrieves context documents and streams an bounded answer back from the LLM."""
    context = retrieve_context(question)
    
    # Strict prompt layout to prevent hallucination out-of-bounds
    system_prompt = (
        "You are a helpful assistant. Answer the user's question ONLY using the provided context below. "
        "If the answer cannot be found in the context, say 'I do not know based on the provided documents.' "
        "Do not use external knowledge. Cite the source of every fact as [Source: file, page N], "
        "copied from the context block it came from.\n\n"
        f"Context:\n{context}\n"
    )
    
    full_prompt = f"{system_prompt}\nQuestion: {question}\nAnswer:"
    
    # Reusing your working streaming utility from Phase 1
    return query_ollama(full_prompt)

if __name__ == "__main__":
    # RRF self-check: an item ranked well in both lists beats one ranked first in only one
    a, b, c = ("a", {}), ("b", {}), ("c", {})
    assert rrf([[a, b], [b, c]])[0] == b

    user_question = "What does the Northwind Health Plus plan cover?"
    for mode in ["dense", "hybrid", "hybrid_rerank"]:
        print(f"\n== {mode} ==")
        for doc, meta in search(user_question, mode):
            print(Path(meta["source"]).name, meta.get("page_number"), "|", doc[:80].replace("\n", " "))
    print("\nStreaming answer from LLM:")
    for token in query_rag(user_question):
        print(token, end='', flush=True)
