import json
import os
import pandas as pd


CSV_PATH = "data/articles/urdu-news-dataset-sports.csv"   
CHUNKS_OUT = "data/chunks.json"
CHUNK_SIZE = 200   
OVERLAP = 50       
def load_csv(path):
    if not os.path.exists(path):
        print(f"CSV not found at '{path}'.")
        print("Put your CSV file at data/articles/urdu-news-dataset-sports.csv and run again.")
        exit(1)

    df = pd.read_csv(path, encoding="cp1256", on_bad_lines="skip")
    print(f"Loaded {len(df)} rows from {path}")
    print(f"Columns found: {list(df.columns)}")

    # Verify required columns exist
    required = ["Date", "Headline", "News Text"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"ERROR: Missing columns: {missing}")
        print(f"Your columns: {list(df.columns)}")
        exit(1)

    # Drop rows with empty content
    before = len(df)
    df = df.dropna(subset=["News Text"])
    df = df[df["News Text"].str.strip() != ""]
    print(f"Dropped {before - len(df)} empty rows. Remaining: {len(df)}")

    return df


def build_articles(df):
    """Convert dataframe rows into article dicts."""
    df = df.reset_index(drop=True)
    articles = []
    for i, row in df.iterrows():
        article_id = f"article_{i:04d}"

        articles.append({
            "id": article_id,
            "title": str(row["Headline"]).strip() if pd.notna(row["Headline"]) else "",
            "content": str(row["News Text"]).strip(),
            "date": str(row["Date"]).strip() if pd.notna(row["Date"]) else "",
            "category": str(row["Category"]).strip() if pd.notna(row["Category"]) else "",
            "url": str(row["URL"]).strip() if pd.notna(row["URL"]) else "",
            "source": str(row["Source"]).strip() if pd.notna(row["Source"]) else ""
        })
    return articles


def chunk_article(article, chunk_size=CHUNK_SIZE, overlap=OVERLAP):
    """
    Split one article into overlapping passages.
    Overlap avoids cutting answers across chunk boundaries.
    """
    words = article["content"].split()
    step = chunk_size - overlap
    chunks = []
    chunk_num = 0

    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if len(chunk_words) < 30:   # skip tiny tail chunks
            continue
        chunks.append({
            "chunk_id": f"{article['id']}_chunk_{chunk_num:03d}",
            "article_id": article["id"],
            "title": article["title"],
            "category": article["category"],
            "date": article["date"],
            "text": " ".join(chunk_words)
        })
        chunk_num += 1

    return chunks


def main():
    df = load_csv(CSV_PATH)
    articles = build_articles(df)
    print(f"Built {len(articles)} article objects")

    all_chunks = []
    for article in articles:
        all_chunks.extend(chunk_article(article))
        if len(all_chunks) >= 10000:
            print("Reached 10k chunks, stopping...")
            break

    print(f"Created {len(all_chunks)} chunks "
          f"(chunk_size={CHUNK_SIZE}, overlap={OVERLAP})")

    os.makedirs("data", exist_ok=True)
    with open(CHUNKS_OUT, "w", encoding="cp1256") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    print(f"Saved to {CHUNKS_OUT}")

    # Show a sample
    if all_chunks:
        s = all_chunks[0]
        print(f"\nSample chunk:")
        print(f"  chunk_id : {s['chunk_id']}")
        print(f"  title    : {s['title']}")
        print(f"  category : {s['category']}")
        print(f"  text     : {s['text'][:150]}...")


if __name__ == "__main__":
    main()
    