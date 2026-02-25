from retrieval.dense_retriever import retrieve as dense_retrieve
from retrieval.sparse_retriever import retrieve as sparse_retrieve


def retrieve(query: str, top_k: int = 5, alpha: float = 0.6) -> list:
    dense_results = dense_retrieve(query, top_k=20)
    sparse_results = sparse_retrieve(query, top_k=20)

    scores = {}
    chunks_map = {}

    # Dense contribution
    for rank, r in enumerate(dense_results):
        key = r["text"]   # stable key

        scores[key] = scores.get(key, 0) + alpha * (1 / (rank + 1))
        chunks_map[key] = r

    # Sparse contribution
    for rank, r in enumerate(sparse_results):
        key = r["text"]

        scores[key] = scores.get(key, 0) + (1 - alpha) * (1 / (rank + 1))
        chunks_map[key] = r

    # Sort by fused score
    sorted_keys = sorted(scores, key=scores.get, reverse=True)[:top_k]

    return [
        {
            "text": chunks_map[k]["text"],
            "metadata": chunks_map[k]["metadata"],
            "score": scores[k]
        }
        for k in sorted_keys
    ]


if __name__ == "__main__":
    query = "What are the major risk factors for Apple?"
    results = retrieve(query, top_k=5)

    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Company: {r['metadata']['company']} | Year: {r['metadata']['year']}")
        print(r["text"][:300])