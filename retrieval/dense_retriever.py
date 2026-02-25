from sentence_transformers import SentenceTransformer
from database.vector_store import search, load_index

MODEL_NAME = "all-MiniLM-L6-v2"
model = SentenceTransformer(MODEL_NAME)

load_index()

def retrieve(query: str, top_k: int = 5) -> list:
    query_embedding = model.encode(
        query,
        normalize_embeddings=True
    )
    results = search(query_embedding, top_k=top_k)
    return results

if __name__ == "__main__":
    query = "What are the major risk factors for Apple?"
    results = retrieve(query, top_k=5)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Company: {r['metadata']['company']} | Year: {r['metadata']['year']}")
        print(r['text'][:300])