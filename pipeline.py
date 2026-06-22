# pipeline.py — Full end-to-end interactive RAG pipeline
#
# Prerequisites (run in order):
#   1. python chunking.py
#   2. python build_queries.py
#   3. python normalization/normalize.py
#   4. python embeddings/embed.py
#   5. python retrieval/retrieve.py
#   6. python evaluation/evaluate.py
#
# Then run this for interactive demo:
#   python pipeline.py
#
# Every response is saved to: output/responses.json
# Open that file to read Urdu text correctly.

import os
import sys
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from normalization.normalize import (
    normalize_query,
    transliterate_query,
    load_normalization_map
)
from embeddings.embed import load_index, load_model
from retrieval.retrieve import baseline_retrieve, improved_retrieve
from generation.generate import get_client, generate_answer


# ─────────────────────────────────────────────
# JSON Output
# ─────────────────────────────────────────────

RESPONSES_PATH = "output/responses.json"

def load_responses():
    """Load existing responses file, or start fresh."""
    if os.path.exists(RESPONSES_PATH):
        with open(RESPONSES_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def save_response(response_record):
    """Append one response record to the JSON file."""
    os.makedirs("output", exist_ok=True)
    responses = load_responses()
    responses.append(response_record)
    with open(RESPONSES_PATH, "w", encoding="utf-8") as f:
        json.dump(responses, f, ensure_ascii=False, indent=2)
    print(f"  Saved to {RESPONSES_PATH} (total: {len(responses)} responses)")


# ─────────────────────────────────────────────
# Load Components
# ─────────────────────────────────────────────

def load_everything():
    """Load all components once at startup."""
    print("=" * 50)
    print("  Loading RAG Pipeline...")
    print("=" * 50)

    model = load_model()
    index, chunks = load_index()
    normalization_map = load_normalization_map()

    if not normalization_map:
        print("WARNING: Normalization map not found. Improved system will skip Lex-Var.")

    try:
        client = get_client()
        generation_available = True
        print("OpenAI client ready.")
    except ValueError as e:
        print(f"WARNING: {e}")
        print("Running in retrieval-only mode (no answer generation).")
        client = None
        generation_available = False

    print("\nAll components loaded. Ready.\n")
    return model, index, chunks, normalization_map, client, generation_available


# ─────────────────────────────────────────────
# Query Runner
# ─────────────────────────────────────────────

def run_query(query, mode, model, index, chunks, normalization_map,
              client, generation_available, top_k=5):
    """
    Run one query through the selected system.
    Prints ASCII-safe info to terminal.
    Saves full Urdu content to JSON.
    """
    print(f"\n{'─'*50}")
    print(f"Query  : {query}")
    print(f"Mode   : {mode}")

    normalized = query
    urdu_script = query

    # ── Retrieval ──────────────────────────────────────
    if mode == "baseline":
        retrieved = baseline_retrieve(query, model, index, chunks, top_k)
        print("Query used for retrieval: (no preprocessing)")
    else:
        retrieved, normalized, urdu_script = improved_retrieve(
            query, model, index, chunks, normalization_map, top_k
        )
        print(f"Normalized : {normalized}")
        print(f"Urdu script: (see JSON output)")

    # ── Terminal: show scores and chunk IDs only ───────
    print(f"\nTop {len(retrieved)} retrieved passages:")
    for i, r in enumerate(retrieved):
        print(f"  [{i+1}] score={r['score']:.3f} | chunk_id={r.get('chunk_id', '?')}")

    # ── Answer Generation ──────────────────────────────
    answer = None
    if generation_available and client:
        print("\nGenerating answer...")
        #display_query = urdu_script if mode == "improved" else query
        answer = generate_answer(query, retrieved[:3], client)
        print(f"Answer: {answer}")
    else:
        print("\n(Answer generation not available — set OPENAI_API_KEY)")

    # ── Build and save full record to JSON ─────────────
    record = {
        "timestamp": datetime.now().isoformat(),
        "mode": mode,
        "query": {
            "original": query,
            "normalized": normalized,
            "urdu_script": urdu_script
        },
        "retrieved_passages": [
            {
                "rank": i + 1,
                "score": round(r["score"], 4),
                "chunk_id": r.get("chunk_id", ""),
                "title": r.get("title", ""),
                "text": r.get("text", "")
            }
            for i, r in enumerate(retrieved)
        ],
        "answer": answer
    }
    save_response(record)
    print(f"{'─'*50}")
    return record


# ─────────────────────────────────────────────
# Interactive Demo
# ─────────────────────────────────────────────

def interactive_demo(model, index, chunks, normalization_map,
                     client, generation_available):
    print("=" * 50)
    print("  Interactive RAG Demo")
    print(f"  Responses saved to: {RESPONSES_PATH}")
    print("  Type 'quit' to exit")
    print("=" * 50)

    while True:
        print()
        query = input("Enter Roman Urdu query: ").strip()
        if not query or query.lower() == "quit":
            print("Exiting.")
            break

        mode_input = input("Mode — (1) Baseline  (2) Improved  (3) Both: ").strip()

        if mode_input == "1":
            modes = ["baseline"]
        elif mode_input == "2":
            modes = ["improved"]
        elif mode_input == "3":
            modes = ["baseline", "improved"]
        else:
            print("Invalid choice. Using improved.")
            modes = ["improved"]

        for mode in modes:
            run_query(
                query, mode,
                model, index, chunks, normalization_map,
                client, generation_available,
                top_k=5
            )


# ─────────────────────────────────────────────
# Batch Demo
# ─────────────────────────────────────────────

def batch_demo(model, index, chunks, normalization_map,
               client, generation_available):
    if not os.path.exists("data/queries.json"):
        print("data/queries.json not found.")
        return

    with open("data/queries.json", "r", encoding="utf-8") as f:
        queries = json.load(f)

    print(f"\nRunning batch demo on {len(queries)} queries...")
    print(f"All responses will be saved to: {RESPONSES_PATH}\n")

    for q in queries:
        run_query(
            q["query"], "improved",
            model, index, chunks, normalization_map,
            client, generation_available,
            top_k=5
        )

    print(f"\nBatch complete. Open {RESPONSES_PATH} to read all responses.")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

if __name__ == "__main__":
    model, index, chunks, normalization_map, client, generation_available = load_everything()

    print("Choose mode:")
    print("  1. Interactive (type your own queries)")
    print("  2. Batch (run all saved queries)")
    choice = input("Enter 1 or 2: ").strip()

    if choice == "2":
        batch_demo(model, index, chunks, normalization_map, client, generation_available)
    else:
        interactive_demo(model, index, chunks, normalization_map, client, generation_available)