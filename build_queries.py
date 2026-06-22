# Step 2: build_queries.py (synthetic version with GPT)
#
# Auto-generates Roman Urdu queries from each chunk using OpenAI GPT.
# Run: python build_queries.py
#
# Requires: pip install openai
# Set env var: export OPENAI_API_KEY="sk-..."

import json
import os
import time
from openai import OpenAI

CHUNKS_PATH = "data/chunks.json"
QUERIES_PATH = "data/queries.json"
QUERIES_PER_CHUNK = 1


client = OpenAI()


def load_chunks():
    if not os.path.exists(CHUNKS_PATH):
        print(f"{CHUNKS_PATH} not found. Run chunking.py first.")
        exit(1)
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")
    return chunks


def generate_queries(chunk, n=QUERIES_PER_CHUNK):
    """
    Generate `n` faithful, diverse, ground-truth Roman Urdu queries for a chunk.

    Raises on malformed/insufficient output instead of silently returning a
    short list, so the caller's except-block can skip the chunk and leave it
    eligible for retry on the next resume (it never gets falsely marked done).
    """
    system_prompt = (
        "You are creating evaluation data for a Roman Urdu RAG (retrieval) system. "
        "Your queries become GROUND TRUTH labels, so accuracy and faithfulness "
        "matter more than style. QUERIES MUST BE IN ROMAN URDU, and must be answerable from the chunk text. "
    )

    prompt = f"""Yeh ek chunk hai jo sirf ~200 words ka hai, kisi bare article ka hissa. Iss chunk se {n} alag-alag search queries banayein jo sirf isi chunk ki maloomat se answer ho sakein.

Rules:
1. Each query must be based only on the specific content of this chunk. Avoid generic or overly broad queries that could also match other chunks (e.g., avoid questions like "What is X?" if X is just the general topic and is discussed in multiple chunks).

2. Write all queries in Roman Urdu, the way an ordinary user would type them into a search bar (short, natural, sometimes slightly informal, and similar to real user searches).

3. Assume the user does not know the information comes from an article. The query should simply seek information directly, without references such as "this article," "this text," "this paragraph," or similar phrases.

4. The {n} queries should be distinct from one another, using different angles, wording, or specific facts/entities from the chunk. Avoid duplicates or near-duplicates.

5. Use only facts that are explicitly present in the chunk. Do not assume, infer, or invent any information that is not stated in the text.

6. If the beginning or end of the chunk appears incomplete or cut off, do not generate queries about that incomplete portion. Focus only on information that is fully contained and clearly presented in the chunk.

Title: {chunk['title']}
Chunk text:
{chunk['text']}

Respond with ONLY a JSON array of {n} strings, nothing else. Example format:
["query 1 text", "query 2 text"]"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=300,
        temperature=0.7,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code fences if the model adds them despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.lower().startswith("json"):
            raw = raw[4:].strip()

    try:
        questions = json.loads(raw)
        if not isinstance(questions, list):
            raise ValueError("Expected a JSON list")
    except (json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(f"Could not parse model output as JSON list: {raw[:200]!r} ({e})")

    # Clean + de-duplicate (case-insensitive) while preserving order
    seen = set()
    cleaned = []
    for q in questions:
        q = str(q).strip().strip('"').strip()
        if q and q.lower() not in seen:
            seen.add(q.lower())
            cleaned.append(q)

    if len(cleaned) < n:
        raise RuntimeError(f"Only got {len(cleaned)}/{n} valid unique queries")

    return cleaned[:n]


def build_synthetic_queries(chunks):
    """
    Generate synthetic queries for all chunks.
    Resumes from existing queries.json if interrupted.
    """
    queries = []

    # Resume if partially done
    if os.path.exists(QUERIES_PATH):
        with open(QUERIES_PATH, "r", encoding="utf-8") as f:
            queries = json.load(f)
        done_ids = {q["relevant_chunk_id"] for q in queries}
        chunks = [c for c in chunks if c["chunk_id"] not in done_ids]
        print(f"Resuming — {len(queries)} queries already saved, {len(chunks)} chunks remaining.")

    total_chunks = len(chunks)
    for i, chunk in enumerate(chunks, 1):
        print(f"  [{i}/{total_chunks}] {chunk['chunk_id']} — {chunk['title'][:50]}")
        try:
            questions = generate_queries(chunk, n=QUERIES_PER_CHUNK)
            for q in questions:
                query_id = f"q_{len(queries) + 1:03d}"
                queries.append({
                    "query_id": query_id,
                    "query": q,
                    "relevant_chunk_id": chunk["chunk_id"]
                })
            save_queries(queries)
            time.sleep(0.1)  # small delay to avoid rate limiting
        except Exception as e:
            print(f"    ERROR on {chunk['chunk_id']}: {e}")
            continue

    return queries


def save_queries(queries):
    os.makedirs("data", exist_ok=True)
    with open(QUERIES_PATH, "w", encoding="utf-8") as f:
        json.dump(queries, f, ensure_ascii=False, indent=2)


def show_stats(queries):
    print(f"\n=== Generation Stats ===")
    print(f"  Total queries generated: {len(queries)}")
    if queries:
        print(f"  Sample query: {queries[0]['query']}")


if __name__ == "__main__":
    # Check API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY='sk-...'")
        exit(1)

    chunks = load_chunks()
    queries = build_synthetic_queries(chunks)
    show_stats(queries)
    print(f"\nDone. {len(queries)} queries saved to {QUERIES_PATH}")