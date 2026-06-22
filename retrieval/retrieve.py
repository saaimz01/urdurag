# Step 5: retrieval/retrieve.py
#
# Two systems:
#   Baseline : raw Roman Urdu query → LaBSE → FAISS
#   Improved : Roman Urdu → Lex-Var → transliterate → LaBSE → FAISS
#
# Run: python retrieval/retrieve.py

import json
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# Add project root to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from normalization.normalize import normalize_query, transliterate_query, load_normalization_map
from embeddings.embed import load_index, load_model


# ─────────────────────────────────────────────
# Core Retrieval
# ─────────────────────────────────────────────

def retrieve(query_text, model, index, chunks, top_k=5):
    """
    Embed query_text and return top_k most similar chunks.
    query_text should be in whatever script the index was built with
    (Urdu script for improved system, Roman Urdu for baseline).
    """
    query_vec = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype(np.float32)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks):
            results.append({
                "chunk_id": chunks[idx]["chunk_id"],
                "title": chunks[idx]["title"],
                "text": chunks[idx]["text"],
                "score": float(score)
            })
    return results


# ─────────────────────────────────────────────
# Baseline System
# ─────────────────────────────────────────────

def baseline_retrieve(roman_urdu_query, model, index, chunks, top_k=5):
    """
    Feed raw Roman Urdu directly into LaBSE — no preprocessing.
    This is the baseline we compare against.
    """
    return retrieve(roman_urdu_query, model, index, chunks, top_k)


# ─────────────────────────────────────────────
# Improved System
# ─────────────────────────────────────────────

def improved_retrieve(roman_urdu_query, model, index, chunks,
                      normalization_map, top_k=5):
    """
    Full pipeline:
      1. Normalize spelling with Lex-Var map
      2. Transliterate to Urdu script
      3. Embed and retrieve
    """
    # Step 1: Lex-Var normalization
    normalized = normalize_query(roman_urdu_query, normalization_map)

    # Step 2: Transliterate to Urdu script
    urdu_script = transliterate_query(normalized)

    # Step 3: Retrieve
    results = retrieve(urdu_script, model, index, chunks, top_k)

    return results, normalized, urdu_script


# ─────────────────────────────────────────────
# Batch Run — all queries
# ─────────────────────────────────────────────

def run_baseline_all(queries, model, index, chunks, top_k=5):
    print(f"\nBaseline retrieval on {len(queries)} queries...")
    all_results = []
    for q in queries:
        retrieved = baseline_retrieve(q["query"], model, index, chunks, top_k)
        all_results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "relevant_chunk_id": q["relevant_chunk_id"],
            "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
            "retrieved_scores": [r["score"] for r in retrieved]
        })
    return all_results


def run_improved_all(queries, model, index, chunks, normalization_map, top_k=5):
    print(f"\nImproved retrieval on {len(queries)} queries...")
    all_results = []
    for q in queries:
        retrieved, normalized, urdu_script = improved_retrieve(
            q["query"], model, index, chunks, normalization_map, top_k
        )
        all_results.append({
            "query_id": q["query_id"],
            "query": q["query"],
            "normalized_query": normalized,
            "urdu_script_query": urdu_script,
            "relevant_chunk_id": q["relevant_chunk_id"],
            "retrieved_chunk_ids": [r["chunk_id"] for r in retrieved],
            "retrieved_scores": [r["score"] for r in retrieved]
        })
    return all_results


def save_results(results, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(results)} results to {path}")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    # Load queries
    if not os.path.exists("data/queries.json"):
        print("data/queries.json not found. Run build_queries.py first.")
        exit(1)

    with open("data/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries")

    # Load model and index
    model = load_model()
    index, chunks = load_index()

    # Load Lex-Var normalization map
    normalization_map = load_normalization_map()
    if not normalization_map:
        print("WARNING: Normalization map empty. Run normalization/normalize.py first.")
        print("Improved system will run without Lex-Var normalization.")

    # Run both systems
    baseline_results = run_baseline_all(queries, model, index, chunks, top_k=5)
    improved_results = run_improved_all(queries, model, index, chunks,
                                        normalization_map, top_k=5)

    # Save results
    save_results(baseline_results, "retrieval/baseline_results.json")
    save_results(improved_results, "retrieval/improved_results.json")

    print("\nRetrieval done. Run evaluation/evaluate.py next.")