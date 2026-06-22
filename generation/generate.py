# Step 7: generation/generate.py
#
# Takes retrieved chunks + query → sends to OpenAI → returns answer.

import os
from openai import OpenAI


# ─────────────────────────────────────────────
# Client Setup
# ─────────────────────────────────────────────

def get_client():
    """
    OpenAI uses the OpenAI SDK with a different base_url.
    Never hardcode your API key — always use an environment variable.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable not set.\n"
            "Windows : set OPENAI_API_KEY=your_key_here\n"
            "Mac/Linux: export OPENAI_API_KEY=your_key_here"
        )
    return OpenAI(
        api_key=api_key
    )


# ─────────────────────────────────────────────
# Prompt Builder
# ─────────────────────────────────────────────

def build_prompt(query, retrieved_chunks):
    """
    Build the RAG prompt.
    We give the model the retrieved Urdu passages and ask it to answer
    the question using ONLY those passages.

    The query can be in Roman Urdu — GPT understands it.
    The context passages are in Urdu script (from your articles).
    """
    context_parts = []
    for i, chunk in enumerate(retrieved_chunks):
        context_parts.append(
            f"Passage {i+1} (from: {chunk.get('title', 'Unknown')}):\n{chunk['text']}"
        )
    context = "\n\n".join(context_parts)

    prompt = f"""You are a helpful assistant that answers questions using the provided Urdu passages.

Instructions:
- Answer using the information in the passages below and your own knowledge.
- Keep the answer relevant.
- Answer in Roman Urdu.

Passages:
{context}

Question: {query}

Answer:"""
    return prompt


# ─────────────────────────────────────────────
# Generation
# ─────────────────────────────────────────────

def generate_answer(query, retrieved_chunks, client=None, model="gpt-4o-mini"):
    """
    Generate an answer for the query given retrieved chunks.

    Args:
        query          : the original query (Roman Urdu or any form)
        retrieved_chunks: list of dicts with 'text' and 'title' keys
        client         : OpenAI client (created if None)
        model          : GPT model name ("gpt-4o-mini" is free tier)

    Returns:
        answer string
    """
    if client is None:
        client = get_client()

    if not retrieved_chunks:
        return "No relevant passages found to answer this question."

    prompt = build_prompt(query, retrieved_chunks)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that answers questions based on provided text passages. You answer in Roman Urdu, dont answer in Urdu script. If the answer is not in the passages, use your own knowledge but don't say 'Maalum nhi'."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=512,
            temperature=0.1,   # low temperature = more factual, less creative
            stream=False
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Generation failed: {str(e)}"


def generate_for_all_queries(results, client, top_k_for_generation=3):
    """
    Generate answers for all queries in a results list.
    We use top 3 chunks for generation (more than 3 tends to confuse the model).
    """
    # Load chunk texts (results only have chunk_ids)
    import json
    with open("embeddings/chunks_metadata.json", "r", encoding="utf-8") as f:
        all_chunks = json.load(f)
    chunk_lookup = {c["chunk_id"]: c for c in all_chunks}

    answered = []
    for r in results:
        # Get full chunk objects for top-k
        top_chunk_ids = r["retrieved_chunk_ids"][:top_k_for_generation]
        retrieved_chunks = [
            chunk_lookup[cid] for cid in top_chunk_ids
            if cid in chunk_lookup
        ]

        query = r.get("urdu_script_query") or r.get("query")
        answer = generate_answer(query, retrieved_chunks, client)

        answered.append({
            "query_id": r["query_id"],
            "query": r["query"],
            "answer": answer,
            "relevant_chunk_id": r["relevant_chunk_id"],
            "retrieved_chunk_ids": r["retrieved_chunk_ids"]
        })
        print(f"  {r['query_id']}: {answer[:80]}...")

    return answered


if __name__ == "__main__":
    # Quick test
    client = get_client()

    test_chunks = [
        {
            "title": "Test Article",
            "text": "پاکستان میں مہنگائی کی وجہ روپے کی قدر میں کمی اور درآمدات کی بڑھتی ہوئی قیمتیں ہیں۔"
        }
    ]
    test_query = "Pakistan mein mehngai kyun hai"

    print("Testing OPENAI generation...")
    answer = generate_answer(test_query, test_chunks, client)
    print(f"Query : {test_query}")
    print(f"Answer: {answer}")