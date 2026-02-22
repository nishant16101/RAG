import os
import numpy as np
import pickle
import os
import faiss

EMBEDDINGS_DIR = "data/embeddings"

def build_index():
    embeddings = np.load(f"{EMBEDDINGS_DIR}/embeddings.npy").astype("float32")
    with open(f"{EMBEDDINGS_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    print(f"Faiss index for {len(embeddings)} vectors")
    dimension= embeddings.shape[1]

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)
    faiss.write_index(index, f"{EMBEDDINGS_DIR}/faiss.index")
    print("Done")


def search(query_embedding:np.ndarray,top_k:int=5):
    index = faiss.read_index(f"{EMBEDDINGS_DIR}/faiss.index")
    with open(f"{EMBEDDINGS_DIR}/chunks.pkl", "rb") as f:
        chunks = pickle.load(f)

    q = query_embedding.astype("float32").reshape(1,-1)
    faiss.normalize_L2(q)
    scores,indices = index.search(q,top_k)

    results = []
    for score ,idx in zip(scores[0],indices[0]):
        if idx != -1:
            results.append({"chunk": chunks[idx], "score": float(score)})


        return results

if __name__ == "__main__":
    build_index()