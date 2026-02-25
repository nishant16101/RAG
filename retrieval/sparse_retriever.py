import json
import numpy as np
from rank_bm25 import BM25Okapi

CHUNKS_PATH = "data/embeddings/chunks.json"


def load_bm25():
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    tokenized_corpus = [
        chunk["text"].lower().split()
        for chunk in chunks
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    return bm25, chunks


# Load once (like FAISS)
bm25, chunks = load_bm25()


def retrieve(query: str, top_k: int = 5) -> list:
    tokens = query.lower().split()

    scores = bm25.get_scores(tokens)

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for idx in top_indices:
        results.append({
            "text": chunks[idx]["text"],
            "metadata": chunks[idx]["metadata"],
            "score": float(scores[idx])
        })

    return results


if __name__ == "__main__":
    query = "What are the major risk factors for Apple?"

    results = retrieve(query, top_k=5)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Company: {r['metadata']['company']} | Year: {r['metadata']['year']}")
        print(r["text"][:300])