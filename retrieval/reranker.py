from sentence_transformers import CrossEncoder
from retrieval.hybrid_retriever import retrieve as hybrid_retrieve

MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
model = CrossEncoder(MODEL_NAME)


def rerank(query: str, top_k: int = 5) -> list:
    candidates = hybrid_retrieve(query, top_k=20)

    # Prepare query-document pairs
    pairs = [(query, r["text"]) for r in candidates]

    scores = model.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    return [
        {
            "text": r["text"],
            "metadata": r["metadata"],
            "score": float(score)
        }
        for score, r in ranked[:top_k]
    ]


if __name__ == "__main__":
    query = "What are the major risk factors for Apple?"
    results = rerank(query, top_k=5)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Company: {r['metadata']['company']} | Year: {r['metadata']['year']}")
        print(r["text"][:300])