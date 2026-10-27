""" 
Loads the PDF files from the specified directory and extracts text from them.
"""

import os
import glob
from pypdf import PdfReader
import chromadb
from embeddings import get_embedding_function, ROOT, CHROMA_PATH, COLLECTION

def extract_text_from_pdfs(directory_path):
    """Load PDF files from the specified directory and extract text from them."""
    documents = []
    pdf_files = glob.glob(os.path.join(directory_path, "*.pdf"))

    if not pdf_files:
        print(f"No PDF files found in: {directory_path}")
        return documents

    for pdf_file in pdf_files:
        print(f"Extracting text from: {pdf_file}")
        try:
            reader = PdfReader(pdf_file)
            for page_num, page in enumerate(reader.pages):
                text = page.extract_text()
                if text and text.strip():
                    documents.append({
                        "text":text,
                        "metadata": {
                            "source": pdf_file,
                            "page_number": page_num + 1
                        }
                    })
        except Exception as e:
            print(f"Error reading {pdf_file}: {e}")

    return documents

def load_markdown(directory_path):
    """Load .md files as one document each (no pages)."""
    documents = []
    for md_file in glob.glob(os.path.join(directory_path, "*.md")):
        print(f"Loading: {md_file}")
        with open(md_file, encoding="utf-8") as f:
            documents.append({"text": f.read(), "metadata": {"source": md_file}})
    return documents

# ~380 words is roughly 500 tokens for English text
def chunk_text(documents:list[dict], chunk_size:int=380, chunk_overlap:int=50)->list[dict]:
    chunks = []
    for doc in documents:
        words = doc['text'].split()
        for i in range(0, len(words), chunk_size - chunk_overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)

            #metadata
            metadata = doc["metadata"].copy()
            metadata["chunk_index"] = i // (chunk_size - chunk_overlap) + 1

            chunks.append({
                "text": chunk_text,
                "metadata": metadata
            })
    return chunks

def main(directory_path: str = str(ROOT / "data")):
    documents = extract_text_from_pdfs(directory_path) + load_markdown(directory_path)
    chunks = chunk_text(documents)

    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    # Rebuild from scratch so re-running ingest doesn't hit duplicate ids or keep stale chunks
    if COLLECTION in [c.name for c in chroma_client.list_collections()]:
        chroma_client.delete_collection(COLLECTION)

    collection = chroma_client.create_collection(
        name=COLLECTION,
        embedding_function=get_embedding_function()
    )

    # ids = [f"{chunk['metadata']['source']}_page{chunk['metadata']['page_number']}_chunk{chunk['metadata']['chunk_index']}" for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    documents_text = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]

    # Batches keep each Ollama embedding call under the client's 60s timeout
    batch = 32
    for i in range(0, len(chunks), batch):
        collection.add(
            ids=ids[i:i + batch],
            documents=documents_text[i:i + batch],
            metadatas=metadatas[i:i + batch]
        )
        print(f"Embedded {min(i + batch, len(chunks))}/{len(chunks)}")
    print(f"Added {len(chunks)} chunks to the ChromaDB collection '{COLLECTION}'.")
   

#test
if __name__ == "__main__":
    import sys
    main(*sys.argv[1:2])  # optional folder, e.g. data/poisoned with RAG_COLLECTION set