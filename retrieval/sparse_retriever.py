import pickle
from rank_bm25 import BM25Okapi

CHUNKS_PATH = "data/embeddings/chunks.pkl"

def load_bm25():
    with open(CHUNKS_PATH,"rb") as f:
        chunks = pickle.load(f)
    tokenized = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized)
    return bm25,chunks

bm25, chunks = load_bm25()
def retrieve(query:str,top_k:int=5)->list:
    tokens = query.lower().split()
    scores = bm25.get_scores(tokens)
    top_indices = scores.argsort()[::-1][:top_k]
    return [{"chunk": chunks[i], "score": float(scores[i])} for i in top_indices]

if __name__ == "__main__":
    query = "What are the major risk factors for Apple"
    results = retrieve(query, top_k=5)
    for i, r in enumerate(results):
        print(f"\n--- Result {i+1} (score: {r['score']:.4f}) ---")
        print(f"Company: {r['chunk']['metadata']['company']} | Year: {r['chunk']['metadata']['year']}")
        print(r['chunk']['text'][:300])
