# Step 4: embeddings/embed.py
#
# Embeds all Urdu article chunks using LaBSE and stores them in a FAISS index.
# Run ONCE before retrieval. Takes a few minutes.

import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
#from tqdm import tqdm


MODEL_NAME = "sentence-transformers/LaBSE"
CHUNKS_PATH = "data/chunks.json"
INDEX_PATH = "embeddings/chunks.index"
METADATA_PATH = "embeddings/chunks_metadata.json"


def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        print(f"{CHUNKS_PATH} not found. Run chunking.py first.")
        exit(1)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    #with open("data/chunks_revised.json", "w", encoding="utf-8") as f:
    #   json.dump(chunks, f, ensure_ascii=False, indent=2)
    return chunks


def load_model():
    print(f"Loading {MODEL_NAME}...")
    print("(First run will download ~1.8GB — this is normal)")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.")
    return model


def embed_chunks(chunks, model, batch_size=32):
    """
    Embed all chunks in batches.
    normalize_embeddings=True means we can use inner product as cosine similarity.
    Returns numpy array of shape (num_chunks, 768).
    """
    texts = [c["text"]+" "+c["date"] for c in chunks]
    print(f"Embedding {len(texts)} chunks (batch_size={batch_size})...")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True
    )

    print(f"Embeddings shape: {embeddings.shape}")  # should be (N, 768)
    return embeddings.astype(np.float32)
    #FAISS optimization — FAISS (the index library) is built for float32

def build_index(embeddings):
    """
    Build a FAISS flat index using inner product.
    Since embeddings are normalized, inner product = cosine similarity.
    IndexFlatIP is exact search (no approximation) — fine for <100k chunks.
    """
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    print(f"FAISS index built: {index.ntotal} vectors, dim={dim}")
    return index


def save_index(index, chunks):
    os.makedirs("embeddings", exist_ok=True)
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Index saved to {INDEX_PATH}")
    print(f"Metadata saved to {METADATA_PATH}")


def load_index():
    """Load the saved index and metadata. Called from retrieve.py."""
    if not os.path.exists(INDEX_PATH):
        print(f"Index not found at {INDEX_PATH}. Run embeddings/embed.py first.")
        exit(1)
    index = faiss.read_index(INDEX_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded index: {index.ntotal} vectors")
    return index, chunks


if __name__ == "__main__":
    chunks = load_chunks()
    model = load_model()
    embeddings = embed_chunks(chunks, model, batch_size=32)
    index = build_index(embeddings)
    save_index(index, chunks)
    print("\nEmbedding complete. Ready for retrieval.")
    #print(chunks[0])