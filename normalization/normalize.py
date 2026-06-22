# normalization/normalize.py
#
# Uses UrduPhone from the Lex-Var repo for phonetic normalization.
# UrduPhone encodes Roman Urdu words into phonetic codes so that
# spelling variants like "kaisay", "kaise", "kaisy" all get the
# same code and are treated as the same word during retrieval.
#
# Setup:
#   git clone https://github.com/abdulrafae/normalization
#   The file we need is: normalization/UrduPhone/UrduPhone.py
#
# Run: python normalization/normalize.py

import json
import os
import sys


# ─────────────────────────────────────────────────────────────
# Setup: import UrduPhone from the cloned repo
# ─────────────────────────────────────────────────────────────

URDUPHONE_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "lexvar_repo", "UrduPhone"
)

def load_urduphone():
    """
    Import UrduPhone.py from the cloned Lex-Var repo.
    Returns the phonetics module or None if not found.
    """
    if not os.path.exists(URDUPHONE_PATH):
        print(f"UrduPhone not found at: {URDUPHONE_PATH}")
        print("Clone the repo first:")
        print("  git clone https://github.com/abdulrafae/normalization normalization/lexvar_repo")
        return None

    sys.path.insert(0, URDUPHONE_PATH)
    try:
        import UrduPhone as phonetics #type:ignore
        print("UrduPhone loaded successfully.")
        return phonetics
    except ImportError as e:
        print(f"Failed to import UrduPhone: {e}")
        return None


# ─────────────────────────────────────────────────────────────
# Part A: Build normalization map using UrduPhone
#
# How it works:
#   1. Take all unique words across all queries
#   2. Compute UrduPhone encoding for each word
#   3. Group words with the same encoding (they are spelling variants)
#   4. Pick the most frequent word in each group as the canonical form
#   5. Build a map: variant → canonical
#
# Example:
#   "kaisay" → encoding "KS"
#   "kaise"  → encoding "KS"   (same!)
#   "kaisy"  → encoding "KS"   (same!)
#   canonical = most frequent among these three
#   map = {"kaisay": "kaise", "kaisy": "kaise"}  (if "kaise" is most frequent)
# ─────────────────────────────────────────────────────────────

def build_normalization_map(queries, phonetics):
    """
    Build a word → canonical_form map using UrduPhone encodings.
    Words with the same UrduPhone code are spelling variants of each other.
    The most frequent one in the query corpus becomes the canonical form.
    """
    # Count word frequencies across all queries
    word_freq = {}
    for q in queries:
        for word in q["query"].lower().split():
            word_freq[word] = word_freq.get(word, 0) + 1

    all_words = list(word_freq.keys())

    # Group words by their UrduPhone encoding
    encoding_groups = {}  # encoding → list of words
    for word in all_words:
        try:
            # UrduPhone crashes on empty string (does term[0] internally)
            # Also skip words with non-ASCII characters (Urdu script mixed in)
            if not word or not word.isascii():
                code = word
            else:
                code = phonetics.UrduPhone(word)
        except Exception:
            code = word  # fallback: treat word as its own code
        
        if code not in encoding_groups:
            encoding_groups[code] = []
        encoding_groups[code].append(word)

    # Build normalization map: variant → canonical (most frequent in group)
    normalization_map = {}
    for code, words in encoding_groups.items():
        if len(words) == 1:
            continue  # no variants, skip
        # Pick the most frequent word as canonical
        canonical = max(words, key=lambda w: word_freq.get(w, 0))
        for word in words:
            if word != canonical:
                normalization_map[word] = canonical

    print(f"UrduPhone groups: {len(encoding_groups)}")
    print(f"Normalization map: {len(normalization_map)} variant words mapped to canonical forms")

    # Show some examples
    shown = 0
    for variant, canonical in normalization_map.items():
        if shown >= 5:
            break
        print(f"  '{variant}' → '{canonical}'")
        shown += 1

    return normalization_map


# ─────────────────────────────────────────────────────────────
# Part B: Apply normalization to a query
# ─────────────────────────────────────────────────────────────

def normalize_query(query, normalization_map):
    """
    Replace spelling variants in a query with their canonical form.
    Words not in the map are left unchanged.

    Example:
      query = "kaisay ho aap"
      map   = {"kaisay": "kaise"}
      result = "kaise ho aap"
    """
    if not normalization_map:
        return query
    words = query.lower().split()
    return " ".join(normalization_map.get(w, w) for w in words)


# ─────────────────────────────────────────────────────────────
# Part C: Transliteration — Roman Urdu → Urdu script
#
# Rule-based, using the UrduPhone homophone mapping table
# from Table 2 of Rafae et al. (the paper you read).
# Longest match first so digraphs (kh, gh, sh...) are matched
# before single characters.
# ─────────────────────────────────────────────────────────────

ROMAN_TO_URDU = [
    # Digraphs first (order matters — longest match)
    ("kh", "خ"), ("gh", "غ"), ("sh", "ش"), ("ch", "چ"),
    ("ph", "پھ"), ("bh", "بھ"), ("th", "تھ"), ("dh", "دھ"),
    ("rh", "ڑ"), ("jh", "جھ"), ("zh", "ژ"),
    ("aa", "آ"), ("ee", "ی"), ("oo", "و"), ("ai", "ے"),
    # Single characters
    ("a", "ا"), ("b", "ب"), ("p", "پ"), ("t", "ت"),
    ("j", "ج"), ("d", "د"), ("r", "ر"), ("z", "ز"),
    ("s", "س"), ("f", "ف"), ("q", "ق"), ("k", "ک"),
    ("g", "گ"), ("l", "ل"), ("m", "م"), ("n", "ن"),
    ("w", "و"), ("v", "و"), ("h", "ہ"), ("y", "ی"),
    ("e", "ے"), ("i", "ی"), ("o", "و"), ("u", "و"),
    ("x", "ز"), ("c", "ک"),
]

def transliterate_word(word):
    """Convert a single Roman Urdu word to Urdu script."""
    word = word.lower()
    result = ""
    i = 0
    while i < len(word):
        matched = False
        for roman, urdu in ROMAN_TO_URDU:
            if word[i:i+len(roman)] == roman:
                result += urdu
                i += len(roman)
                matched = True
                break
        if not matched:
            result += word[i]  # keep digits, punctuation, English as-is
            i += 1
    return result


def transliterate_query(text):
    """
    Transliterate a full Roman Urdu query to Urdu script word by word.
    Purely numeric tokens are kept as-is.
    """
    return " ".join(
        word if word.isdigit() else transliterate_word(word)
        for word in text.split()
    )


# ─────────────────────────────────────────────────────────────
# Full Pipeline: normalize → transliterate
# ─────────────────────────────────────────────────────────────

def preprocess_query(query, normalization_map):
    """
    Full preprocessing pipeline for one query.
    Returns dict with all intermediate forms.
    """
    normalized = normalize_query(query, normalization_map)
    urdu_script = transliterate_query(normalized)
    return {
        "original": query,
        "normalized": normalized,
        "urdu_script": urdu_script
    }


# ─────────────────────────────────────────────────────────────
# Save / Load normalization map
# ─────────────────────────────────────────────────────────────

def save_normalization_map(normalization_map,
                           path="normalization/normalization_map.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(normalization_map, f, ensure_ascii=False, indent=2)
    print(f"Normalization map saved to {path}")


def load_normalization_map(path="normalization/normalization_map.json"):
    if not os.path.exists(path):
        print(f"Normalization map not found at {path}.")
        print("Run normalization/normalize.py first.")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Load queries
    queries_path = "data/queries.json"
    if not os.path.exists(queries_path):
        print(f"{queries_path} not found. Run build_queries.py first.")
        exit(1)

    with open(queries_path, "r", encoding="utf-8") as f:
        queries = json.load(f)
    print(f"Loaded {len(queries)} queries")

    # Load UrduPhone
    phonetics = load_urduphone()
    if phonetics is None:
        print("Continuing without UrduPhone normalization.")
        normalization_map = {}
    else:
        normalization_map = build_normalization_map(queries, phonetics)
        save_normalization_map(normalization_map)

    # Preprocess all queries and save
    processed = []
    for q in queries:
        result = preprocess_query(q["query"], normalization_map)
        processed.append({
            "query_id": q["query_id"],
            "original_query": result["original"],
            "normalized_query": result["normalized"],
            "urdu_script_query": result["urdu_script"],
            "relevant_chunk_id": q["relevant_chunk_id"]
        })

    with open("data/processed_queries.json", "w", encoding="utf-8") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(processed)} processed queries to data/processed_queries.json")

    # Show samples
    print("\nSample outputs:")
    for p in processed[:3]:
        print(f"  Original  : {p['original_query']}")
        print(f"  Normalized: {p['normalized_query']}")
        print(f"  Urdu      : {p['urdu_script_query']}")
        print()