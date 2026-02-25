import os
import numpy as np
import json
import faiss

EMBEDDINGS_DIR = "data/embeddings"

index = None
chunks = None


def build_index():
    embeddings_path = os.path.join(EMBEDDINGS_DIR, "embeddings.npy")
    index_path = os.path.join(EMBEDDINGS_DIR, "faiss.index")

    embeddings = np.load(embeddings_path).astype("float32")

    print(f"Building FAISS index for {len(embeddings)} vectors")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    faiss.write_index(index, index_path)

    print("FAISS index saved successfully.")


def load_index():
    global index, chunks

    index_path = os.path.join(EMBEDDINGS_DIR, "faiss.index")
    chunks_path = os.path.join(EMBEDDINGS_DIR, "chunks.json")

    if not os.path.exists(index_path):
        raise ValueError("FAISS index not found. Run build_index() first.")

    index = faiss.read_index(index_path)

    with open(chunks_path, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print("FAISS index and metadata loaded successfully.")


def search(query_embedding: np.ndarray, top_k: int = 5):
    global index, chunks

    if index is None or chunks is None:
        raise ValueError("Index not loaded. Call load_index() first.")

    q = query_embedding.astype("float32").reshape(1, -1)

    # If embedding was normalized earlier, this is optional
    faiss.normalize_L2(q)

    scores, indices = index.search(q, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "text": chunks[idx]["text"],
                "metadata": chunks[idx]["metadata"],
                "score": float(score)
            })

    return results


if __name__ == "__main__":
    build_index()